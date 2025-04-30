use crate::constants::TARGET_COLUMN;
use crate::preprocessing::REDUNDANT_STRING_VALUE;
use crate::rework::sort_type::SortType;
use crate::rework::{categorical_columns, sort_type};
use polars::prelude::{col, lit, JoinArgs, JoinType, UnionArgs};
use polars_core::datatypes::DataType;
use polars_core::prelude::{NamedFrom, SortMultipleOptions, UniqueKeepStrategy};
use polars_lazy::frame::LazyFrame;
use polars_lazy::prelude::concat;
use std::error::Error;

// TODO: Add fail safe to ensure TARGET_COLUMN doesn't already exist in dataframe

const QUANTILES: [f64; 8] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9];
pub const COUNT_IN: &str = "count_in";
pub const COUNT_OUT: &str = "count_out";
const GINI_IMPURITY_IN_GROUP: &str = "gini_in";
const GINI_IMPURITY_OUT_GROUP: &str = "gini_out";
const TOTAL_IN_GROUP: &str = "total_in_group";
const TOTAL_OUT_GROUP: &str = "total_out_group";
pub const FEATURE_COLUMN_NAME: &str = "FEATURE_COLUMN_NAME";
pub const SORT_TYPE_COL: &str = "SORT_TYPE";
const TEMP_COLUMN_ORDINAL: &str = "temp_ordinal";
pub const SELECTION_COLUMN: &str = "SELECTION_COLUMN";
const NORMALIZED_CHILD_GINI: &str = "NORMALIZED_CHILD_GINI";

fn compute_parent_gini_impurity(lf: &LazyFrame) -> Result<f64, Box<dyn std::error::Error>> {
    let temp_column = "temp";

    let mut grouped = lf
        .clone()
        .select([col(TARGET_COLUMN)])
        .group_by([col(TARGET_COLUMN)])
        .agg([col(TARGET_COLUMN)
            .count()
            .cast(DataType::Float64)
            .alias(temp_column)]);

    let summed_count_df = grouped
        .clone()
        .select([col(temp_column)])
        .sum()
        .collect()
        .unwrap();

    let summed_count = summed_count_df
        .column(temp_column)
        .unwrap()
        .f64()
        .unwrap()
        .get(0)
        .unwrap();

    grouped = grouped.with_column((col(temp_column) / lit(summed_count)).pow(lit(2.0)));
    let summed_square_df = grouped
        .clone()
        .select([col(temp_column)])
        .sum()
        .collect()
        .unwrap();
    let gini_impurity = 1.0
        - summed_square_df
            .column(temp_column)
            .unwrap()
            .f64()
            .unwrap()
            .get(0)
            .unwrap();
    Ok(gini_impurity)
}

fn get_optimal_gini_impurity_for_categorical_column(
    lf: &LazyFrame,
    feature_column: &str,
) -> LazyFrame {
    let lf = pre_process_for_gini(&lf, SortType::Categorical, feature_column);
    let mut grouped_lf = categorical_columns::group_by_for_gini_impurity_categorical(&lf);
    grouped_lf = add_totals_of_in_out_group(&grouped_lf);
    let gini_lf = compute_gini_per_feature(&grouped_lf);
    let normalized_gini_lf = normalize_gini_per_group(&grouped_lf, &gini_lf);

    // Keep only necessary columns and obtain best result:
    let final_lf = normalized_gini_lf
        .select([
            col(FEATURE_COLUMN_NAME),
            col(SORT_TYPE_COL),
            col(SELECTION_COLUMN),
            col(NORMALIZED_CHILD_GINI),
        ])
        .sort([NORMALIZED_CHILD_GINI], SortMultipleOptions::default())
        .limit(1);
    final_lf
}

pub(crate) fn add_zero_count(
    feature_column: &str,
    target_column: &str,
    count_column: &str,
    lf: &LazyFrame,
) -> LazyFrame {
    // Adds a zero if a combination of feature and target doesn't exist for a certain group_by.
    // Get unique combinations of feature and target:
    let combinations = get_unique_combinations(feature_column, target_column, &lf);

    // Add back the count column and fill with zeros:
    let combinations_with_count_lf = combinations
        .join(
            lf.clone()
                .select([col(count_column), col(feature_column), col(target_column)]),
            &[col(feature_column), col(target_column)],
            &[col(feature_column), col(target_column)],
            JoinArgs::new(JoinType::Left),
        )
        .with_columns([col(count_column).fill_null(lit(0))]);

    // Add back the static columns:
    let static_lf = lf
        .clone()
        .select([col("*").exclude([feature_column, count_column])])
        .unique(None, UniqueKeepStrategy::Any);

    let full_lf = combinations_with_count_lf.join(
        static_lf,
        &[col(target_column)],
        &[col(target_column)],
        JoinArgs::new(JoinType::Left),
    );
    full_lf
}

fn get_optimal_gini_impurity_for_column(
    lf: &LazyFrame,
    feature_column: &str,
    sort_type: SortType,
) -> LazyFrame {
    let lf_output: LazyFrame;
    match sort_type {
        SortType::Ordinal => {
            lf_output = get_optimal_gini_impurity_for_ordinal_column(lf, feature_column);
        }
        SortType::Categorical => {
            lf_output = get_optimal_gini_impurity_for_categorical_column(lf, feature_column);
        }
    }
    lf_output
}

fn get_optimal_gini_impurity_for_ordinal_column(lf: &LazyFrame, feature_column: &str) -> LazyFrame {
    let lf = pre_process_for_gini(lf, SortType::Ordinal, feature_column);

    // Gather lazy frames for every quantile:
    let mut lazy_frames: Vec<LazyFrame> = Vec::new();
    for quantile in QUANTILES.iter() {
        lazy_frames.push(group_by_for_single_quantile(&lf, feature_column, *quantile))
    }

    // Combine lazyframes into single, larger lazyframe:
    let mut grouped_lf = concat(
        &lazy_frames,
        UnionArgs {
            rechunk: true,
            to_supertypes: false,
            diagonal: false,
            from_partitioned_ds: false,
            parallel: true,
            maintain_order: false,
        },
    )
    .unwrap();

    grouped_lf = add_totals_of_in_out_group(&grouped_lf);
    let gini_lf = compute_gini_per_feature(&grouped_lf);
    let normalized_gini_lf = normalize_gini_per_group(&grouped_lf, &gini_lf);
    // Keep only necessary columns and obtain best result:
    let final_lf = normalized_gini_lf
        .select([
            col(FEATURE_COLUMN_NAME),
            col(SORT_TYPE_COL),
            col(SELECTION_COLUMN),
            col(NORMALIZED_CHILD_GINI),
        ])
        .sort([NORMALIZED_CHILD_GINI], SortMultipleOptions::default())
        .limit(1);
    final_lf
}

fn group_by_for_single_quantile(lf: &LazyFrame, feature_column: &str, quantile: f64) -> LazyFrame {
    // Add quantile as string:
    let mut quantile_lf = lf.clone().with_column(
        col(feature_column)
            .quantile(lit(quantile), Default::default())
            .cast(DataType::String)
            .alias(SELECTION_COLUMN),
    );
    quantile_lf = quantile_lf
        .with_column(
            col(feature_column)
                .gt_eq(col(feature_column).quantile(lit(quantile), Default::default()))
                .alias(TEMP_COLUMN_ORDINAL),
        )
        .drop([feature_column]);
    let grouped_lf = group_by_for_ordinal_inner(&quantile_lf);
    grouped_lf
}

fn group_by_for_ordinal_inner(lf: &LazyFrame) -> LazyFrame {
    let mut grouped_lf = lf
        // Add count in:
        .clone()
        .group_by([col("*")])
        .agg([col(TARGET_COLUMN)
            .count()
            .alias(COUNT_IN)
            .cast(DataType::Float64)]);

    grouped_lf = add_zero_count(TEMP_COLUMN_ORDINAL, TARGET_COLUMN, COUNT_IN, &grouped_lf);

    // Add count out:
    grouped_lf = grouped_lf
        .with_columns([col(COUNT_IN)
            .sum()
            .over([col(TARGET_COLUMN)])
            .alias("total_per_target")])
        .with_columns([(col("total_per_target") - col(COUNT_IN)).alias(COUNT_OUT)])
        .filter(col(TEMP_COLUMN_ORDINAL))
        .drop(["total_per_target", TEMP_COLUMN_ORDINAL])
        .sort([SELECTION_COLUMN, TARGET_COLUMN], Default::default());
    grouped_lf
}

fn normalize_gini_per_group(grouped_lf: &LazyFrame, gini_lf: &LazyFrame) -> LazyFrame {
    let mut normalized_lf = grouped_lf.clone().join(
        gini_lf.clone(),
        [col(SELECTION_COLUMN)],
        [col(SELECTION_COLUMN)],
        JoinArgs::new(JoinType::Left),
    );

    // Normalize gini_in and gini_out:
    normalized_lf = normalized_lf.with_column(
        ((col(GINI_IMPURITY_IN_GROUP) * col(TOTAL_IN_GROUP))
            / (col(TOTAL_IN_GROUP) + col(TOTAL_OUT_GROUP)))
        .alias(GINI_IMPURITY_IN_GROUP),
    );

    normalized_lf = normalized_lf.with_column(
        ((col(GINI_IMPURITY_OUT_GROUP) * col(TOTAL_OUT_GROUP))
            / (col(TOTAL_IN_GROUP) + col(TOTAL_OUT_GROUP)))
        .alias(GINI_IMPURITY_OUT_GROUP),
    );

    // Get total Gini Impurity of each possible split:
    normalized_lf = normalized_lf.with_column(
        (col(GINI_IMPURITY_IN_GROUP) + col(GINI_IMPURITY_OUT_GROUP)).alias(NORMALIZED_CHILD_GINI),
    );
    normalized_lf
}

fn compute_gini_per_feature(grouped_lf: &LazyFrame) -> LazyFrame {
    let mut gini_lf = grouped_lf.clone().with_column(
        ((col(COUNT_IN) / col(TOTAL_IN_GROUP)).pow(lit(2.0))).alias(GINI_IMPURITY_IN_GROUP),
    );

    gini_lf = gini_lf.with_column(
        ((col(COUNT_OUT) / col(TOTAL_OUT_GROUP)).pow(lit(2.0))).alias(GINI_IMPURITY_OUT_GROUP),
    );

    gini_lf = gini_lf
        .select([
            col(SELECTION_COLUMN),
            col(GINI_IMPURITY_IN_GROUP),
            col(GINI_IMPURITY_OUT_GROUP),
        ])
        .group_by([col(SELECTION_COLUMN)])
        .agg([
            col(GINI_IMPURITY_IN_GROUP).sum(),
            col(GINI_IMPURITY_OUT_GROUP).sum(),
        ])
        .with_columns([
            (lit(1.0) - col(GINI_IMPURITY_IN_GROUP)).alias(GINI_IMPURITY_IN_GROUP),
            (lit(1.0) - col(GINI_IMPURITY_OUT_GROUP)).alias(GINI_IMPURITY_OUT_GROUP),
        ]);
    gini_lf
}

fn add_totals_of_in_out_group(grouped_lf: &LazyFrame) -> LazyFrame {
    let in_group_lf = grouped_lf
        .clone()
        .group_by([col(SELECTION_COLUMN)])
        .agg([col(COUNT_IN).sum().alias(TOTAL_IN_GROUP)]);

    let grouped_lf = grouped_lf.clone().join(
        in_group_lf,
        [col(SELECTION_COLUMN)],
        [col(SELECTION_COLUMN)],
        JoinArgs::new(JoinType::Left),
    );

    let out_group_lf = grouped_lf
        .clone()
        .group_by([col(SELECTION_COLUMN)])
        .agg([col(COUNT_OUT).sum().alias(TOTAL_OUT_GROUP)]);

    let grouped_lf = grouped_lf.clone().join(
        out_group_lf,
        [col(SELECTION_COLUMN)],
        [col(SELECTION_COLUMN)],
        JoinArgs::new(JoinType::Left),
    );
    grouped_lf
}

fn pre_process_for_gini(lf: &LazyFrame, sort_type: SortType, feature_column: &str) -> LazyFrame {
    // After this this step every feature has the same columns:
    let mut lf = lf
        .clone()
        .select([col(feature_column), col(TARGET_COLUMN)])
        .with_columns([
            lit(feature_column).alias(FEATURE_COLUMN_NAME),
            lit(sort_type.as_str()).alias(SORT_TYPE_COL),
        ]);
    if sort_type == SortType::Categorical {
        lf = lf.rename([feature_column], [SELECTION_COLUMN], true);
    }
    lf
}

fn get_unique_combinations(column_1: &str, column_2: &str, lf: &LazyFrame) -> LazyFrame {
    // Get unique values:
    let lf1 = lf
        .clone()
        .select([col(column_1)])
        .unique(None, UniqueKeepStrategy::Any);
    let lf2 = lf
        .clone()
        .select([col(column_2)])
        .unique(None, UniqueKeepStrategy::Any);
    // Cross join the two unique sets
    let combinations = lf1.cross_join(lf2, None);
    combinations
}

pub fn get_best_column_to_split_on(lf: &LazyFrame) -> Result<LazyFrame, Box<dyn Error>> {
    let schema = lf.logical_plan.compute_schema()?;
    let mut lazy_frames: Vec<LazyFrame> = Vec::new();
    for (name, dtype) in schema.iter() {
        if name == TARGET_COLUMN {
            continue;
        }
        let sort_type = sort_type::get_sort_type_for_dtype(dtype);
        lazy_frames.push(get_optimal_gini_impurity_for_column(&lf, name, sort_type));
    }
    let grouped_lf = concat(
        &lazy_frames,
        UnionArgs {
            rechunk: true,
            to_supertypes: false,
            diagonal: false,
            from_partitioned_ds: false,
            parallel: true,
            maintain_order: false,
        },
    )
    .unwrap()
    .sort([NORMALIZED_CHILD_GINI], SortMultipleOptions::default())
    .first();
    Ok(grouped_lf)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::TARGET_COLUMN;
    use crate::preprocessing::REDUNDANT_STRING_VALUE;
    use crate::test_utils::{assert_single_row_df_equal, get_preprocessed_test_dataframe};
    use polars::prelude::{col, lit};
    use polars_core::df;
    use polars_core::utils::Container;
    use polars_lazy::prelude::IntoLazy;

    #[test]
    fn test_iterate_over_columns() -> Result<(), Box<dyn std::error::Error>> {
        unsafe {
            std::env::set_var("POLARS_FMT_MAX_COLS", "100");
        }

        let mut lf = get_preprocessed_test_dataframe();
        lf = lf.drop([TARGET_COLUMN]);
        let target_column = "Pclass";
        lf = lf.rename([target_column], [TARGET_COLUMN], true);

        let collected = get_best_column_to_split_on(&lf)?.collect()?;

        let expected_df = df![
            FEATURE_COLUMN_NAME => &["Fare"],
            SORT_TYPE_COL => &["ordinal"],
            SELECTION_COLUMN => &["21.6792"],
            NORMALIZED_CHILD_GINI => &[0.434657_f64],
        ]?;

        assert_eq!(collected.schema(), expected_df.schema());
        assert_single_row_df_equal(&collected, &expected_df)?;

        Ok(())
    }

    #[test]
    fn test_gini_for_categorical_column() -> Result<(), Box<dyn std::error::Error>> {
        unsafe {
            std::env::set_var("POLARS_FMT_MAX_COLS", "100");
        }

        let mut lf = get_preprocessed_test_dataframe();
        lf = lf.drop([TARGET_COLUMN]);
        let feature_column = "Embarked";
        let target_column = "Pclass";
        lf = lf.rename([target_column], [TARGET_COLUMN], true);

        // END OF PRE-PROCESSING, start of Gini computation:
        let final_lf =
            get_optimal_gini_impurity_for_column(&lf, feature_column, SortType::Categorical);
        let collected = final_lf.collect()?;

        let expected_df = df![
            FEATURE_COLUMN_NAME => &["Embarked"],
            SORT_TYPE_COL => &["categorical"],
            SELECTION_COLUMN => &["C"],
            NORMALIZED_CHILD_GINI => &[0.57038_f64],
        ]?;

        assert_eq!(collected.schema(), expected_df.schema());
        assert_single_row_df_equal(&collected, &expected_df)?;

        Ok(())
    }

    #[test]
    fn test_get_unique_combinations() -> Result<(), Box<dyn std::error::Error>> {
        unsafe {
            std::env::set_var("POLARS_FMT_MAX_COLS", "100");
        }

        let column_1 = "col1";
        let column_2 = "col2";
        let lf = df![
            column_1 => ["A", "B", "A"],
            column_2 => ["X", "Y", "Z"],
        ]?
        .lazy();
        let combinations = get_unique_combinations(column_1, column_2, &lf);
        let collected = combinations.collect()?;
        assert_eq!(collected.len(), 6);
        Ok(())
    }

    #[test]
    fn test_add_zero_count() -> Result<(), Box<dyn std::error::Error>> {
        unsafe {
            std::env::set_var("POLARS_FMT_MAX_COLS", "100");
        }

        let feature_column = "col1";
        let target_column = "col2";
        let count_column = "col3";
        let static_column = "col4";
        let lf = df![
            feature_column => ["A", "B", "A"],
            target_column => ["X", "Y", "Z"],
            count_column => [1, 2, 3],
            static_column => ["s", "s", "s"]
        ]?
        .lazy();

        let full_lf = add_zero_count(feature_column, target_column, count_column, &lf);

        assert_eq!(full_lf.collect()?.len(), 6);

        Ok(())
    }

    #[test]
    fn test_gini_for_ordinal_column() -> Result<(), Box<dyn std::error::Error>> {
        unsafe {
            std::env::set_var("POLARS_FMT_MAX_COLS", "100");
            std::env::set_var("POLARS_FMT_MAX_ROWS", "100");
        }

        let mut lf = get_preprocessed_test_dataframe();
        lf = lf.drop([TARGET_COLUMN]);
        let feature_column = "Fare";
        let target_column = "Pclass";
        lf = lf.rename([target_column], [TARGET_COLUMN], true);

        let sort_type = SortType::Ordinal;

        let final_lf = get_optimal_gini_impurity_for_column(&lf, feature_column, SortType::Ordinal);
        let collected = final_lf.collect()?;
        let expected_df = df![
            FEATURE_COLUMN_NAME => &["Fare"],
            SORT_TYPE_COL => &["ordinal"],
            SELECTION_COLUMN => &["21.6792"],
            NORMALIZED_CHILD_GINI => &[0.434657_f64],
        ]?;

        assert_eq!(collected.schema(), expected_df.schema());
        assert_single_row_df_equal(&collected, &expected_df)?;
        Ok(())
    }

    #[test]
    fn test_parent_gini() -> Result<(), Box<dyn std::error::Error>> {
        unsafe {
            std::env::set_var("POLARS_FMT_MAX_COLS", "100");
        }

        let mut lf = get_preprocessed_test_dataframe();
        lf = lf.filter(col("Embarked").neq(lit(REDUNDANT_STRING_VALUE)));
        lf = lf.drop([TARGET_COLUMN]);
        let feature_column = "Embarked";
        let target_column = "Pclass";
        lf = lf.rename([target_column], [TARGET_COLUMN], true);

        let gini = compute_parent_gini_impurity(&lf)?;
        assert_eq!(gini, 0.5941737597760909);

        Ok(())
    }
}
