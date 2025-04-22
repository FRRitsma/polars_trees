use crate::constants::TARGET_COLUMN;
use polars::prelude::{col, lit, JoinArgs, JoinType};
use polars_core::datatypes::DataType;
use polars_core::frame::DataFrame;
use polars_core::prelude::NamedFrom;
use polars_core::series::Series;
use polars_lazy::frame::LazyFrame;

// TODO: Add fail safe to ensure TARGET_COLUMN doesn't already exist in dataframe
pub fn gini_impurity() {}
pub fn compute_gini_impurity_and_sample_proportion(
    lf: &LazyFrame,
    feature_column: &str,
) -> DataFrame {
    let grouped_lf = old_aggregate_features(lf, feature_column);
    let small_count_column = "count_column";

    println!("{:?}", grouped_lf.clone().collect());

    let total_overall_samples: f64 = grouped_lf
        .clone()
        .select([col(small_count_column)])
        .collect()
        .unwrap()
        .column(small_count_column)
        .unwrap()
        .f64()
        .unwrap()
        .into_iter()
        .flatten() // This removes any None values if you have nulls
        .sum();

    // Step 1: Compute total per Embarked
    let big_count_column = "total_count";
    let totals = grouped_lf
        .clone()
        .group_by([col(feature_column)])
        .agg([col(small_count_column).sum().alias(big_count_column)]);

    // Step 2: Join totals back to original data
    let proportion_column = "proportion";
    let squared_proportion_column = "squared_proportion";
    let final_df = grouped_lf
        .join(
            totals,
            [col(feature_column)],
            [col(feature_column)],
            JoinArgs::default(),
        )
        // Step 3: Compute proportion and squared proportion
        .with_columns([
            (col(small_count_column) / col(big_count_column))
                .pow(lit(2.0))
                .alias(squared_proportion_column),
            (col(big_count_column).cast(DataType::Float64) / lit(total_overall_samples))
                .alias(proportion_column),
        ])
        // Step 4: Group again to get sum of squared proportions (per Embarked)
        .group_by([col(feature_column), col(proportion_column)])
        .agg([(lit(1.0) - col(squared_proportion_column).sum()).alias("gini_impurity")])
        // Step 5: Compute the product of proportion and gini_impurity
        .with_column((col("proportion") * col("gini_impurity")).alias("proportion_gini_product"))
        // Step 6: Select only the columns we want to return
        .select([col(feature_column), col("proportion_gini_product")])
        .collect()
        .unwrap();
    final_df
}

const COUNT_IN_COLUMN: &str = "count_in";
const TOTAL_IN_COLUMN: &str = "total_in";
const COUNT_OUT_COLUMN: &str = "count_out";
const TOTAL_OUT_COLUMN: &str = "total_out";

fn old_aggregate_features(lf: &LazyFrame, feature_column: &str) -> LazyFrame {
    let lf = lf.clone();

    // Step 1: Count per (feature_column, TARGET_COLUMN)
    let grouped_lf = lf
        .clone()
        .select([col(feature_column), col(TARGET_COLUMN)])
        .group_by([col(feature_column), col(TARGET_COLUMN)])
        .agg([col(TARGET_COLUMN)
            .count()
            .alias(COUNT_IN_COLUMN)
            .cast(DataType::Float64)]);

    // Step 2: Total count per feature_column
    let total_per_feature = lf.group_by([col(feature_column)]).agg([col(TARGET_COLUMN)
        .count()
        .alias(TOTAL_IN_COLUMN)
        .cast(DataType::Float64)]);

    // Step 3: Join + compute not_grouping_count
    let with_not_grouping = grouped_lf
        .join(
            total_per_feature,
            [col(feature_column)],
            [col(feature_column)],
            JoinArgs::default(),
        )
        .with_column((col(TOTAL_IN_COLUMN) - col(COUNT_IN_COLUMN)).alias(COUNT_OUT_COLUMN));

    // Step 4: Sum of not_grouping_count per feature_column
    let total_not_grouping_per_feature = with_not_grouping
        .clone()
        .group_by([col(feature_column)])
        .agg([col(COUNT_OUT_COLUMN).sum().alias(TOTAL_OUT_COLUMN)]);

    // Step 5: Final join to add total_count_not_grouping
    let result = with_not_grouping
        .join(
            total_not_grouping_per_feature,
            [col(feature_column)],
            [col(feature_column)],
            JoinArgs::default(),
        )
        .sort([feature_column, TARGET_COLUMN], Default::default());

    result
}

fn compute_parent_gini_and_count_samples(
    lf: &LazyFrame,
) -> Result<(f64, f64), Box<dyn std::error::Error>> {
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
    Ok((gini_impurity, summed_count))
}

fn compute_gini_impurity(lf: &LazyFrame, feature_column: &str) -> LazyFrame {
    let squared_lf = lf.clone().with_columns([
        (col(COUNT_IN_COLUMN) / col(TOTAL_IN_COLUMN))
            .pow(lit(2.0))
            .alias("squared_in_column"),
        (col(COUNT_OUT_COLUMN) / col(TOTAL_OUT_COLUMN))
            .pow(lit(2.0))
            .alias("squared_out_column"),
    ]);
    return squared_lf;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::TARGET_COLUMN;
    use crate::preprocessing::REDUNDANT_STRING_VALUE;
    use crate::test_utils::get_preprocessed_test_dataframe;
    use polars::prelude::{col, lit, GetOutput, JoinArgs, PlSmallStr};
    use polars_core::df;
    use polars_core::prelude::{BooleanChunked, ChunkCompareEq, FillNullStrategy};
    use polars_core::utils::rayon::join;

    #[test]
    fn test_old() -> Result<(), Box<dyn std::error::Error>> {
        unsafe {
            std::env::set_var("POLARS_FMT_MAX_COLS", "100");
        }

        let mut lf = get_preprocessed_test_dataframe();
        lf = lf.filter(col("Embarked").neq(lit(REDUNDANT_STRING_VALUE)));
        lf = lf.drop([TARGET_COLUMN]);
        let feature_column = "Embarked";
        let target_column = "Pclass";
        lf = lf.rename([target_column], [TARGET_COLUMN], true);

        lf = old_aggregate_features(&lf, feature_column);
        lf = compute_gini_impurity(&lf, feature_column);
        // let df = compute_gini_impurity_and_sample_proportion(&mut lf, feature_column);
        println!("{:?}", lf.collect());
        Ok(())
    }

    #[test]
    fn test_debug() -> Result<(), Box<dyn std::error::Error>> {
        unsafe {
            std::env::set_var("POLARS_FMT_MAX_COLS", "100");
        }

        let mut lf = get_preprocessed_test_dataframe();
        lf = lf.filter(col("Embarked").neq(lit(REDUNDANT_STRING_VALUE)));
        lf = lf.drop([TARGET_COLUMN]);
        let feature_column = "Embarked";
        let target_column = "Pclass";
        lf = lf.rename([target_column], [TARGET_COLUMN], true);
        let mut grouped_lf = group_in_and_out(&mut lf, feature_column);
        grouped_lf = add_totals_of_in_out_group(feature_column, &mut grouped_lf);
        let gini_lf = compute_gini_per_feature(feature_column, &mut grouped_lf);

        let normalized_gini_lf = normalize_gini_per_group(feature_column, &grouped_lf, &gini_lf);

        // Get the optimal choice:
        let optimal_choice = normalized_gini_lf.select([col(feature_column), col("normalized_child_gini")]).sort(["normalized_child_gini"], Default::default()).limit(1);

        println!("{:?}", optimal_choice.collect());
        Ok(())
    }

    fn normalize_gini_per_group(feature_column: &str, grouped_lf: &LazyFrame, gini_lf: &LazyFrame)  -> LazyFrame{
        let mut normalized_lf = grouped_lf.clone().join(
            gini_lf.clone(),
            [col(feature_column)],
            [col(feature_column)],
            JoinArgs::new(JoinType::Left),
        );

        // Normalize gini_in and gini_out:
        normalized_lf = normalized_lf.with_column(
            ((col("gini_in") * col("total_in_group"))
                / (col("total_in_group") + col("total_out_group")))
                .alias("gini_in"),
        );

        normalized_lf = normalized_lf.with_column(
            ((col("gini_out") * col("total_out_group"))
                / (col("total_in_group") + col("total_out_group")))
                .alias("gini_out"),
        );

        // Get total Gini Impurity of each possible split:
        normalized_lf =
            normalized_lf.with_column((col("gini_in") + col("gini_out")).alias("normalized_child_gini"));
        normalized_lf
    }

    fn compute_gini_per_feature(feature_column: &str, grouped_lf: &mut LazyFrame) -> LazyFrame {
        let gini_in_column = "gini_in";
        let gini_out_column = "gini_out";
        let mut gini_lf = grouped_lf.clone().with_column(
            ((col("count_in") / col("total_in_group")).pow(lit(2.0))).alias(gini_in_column),
        );

        gini_lf = gini_lf.with_column(
            ((col("count_out") / col("total_out_group")).pow(lit(2.0))).alias(gini_out_column),
        );

        gini_lf = gini_lf
            .select([
                col(feature_column),
                col(gini_in_column),
                col(gini_out_column),
            ])
            .group_by([col(feature_column)])
            .agg([col(gini_in_column).sum(), col(gini_out_column).sum()])
            .with_columns([
                (lit(1.0) - col(gini_in_column)).alias(gini_in_column),
                (lit(1.0) - col(gini_out_column)).alias(gini_out_column),
            ]);
        gini_lf
    }

    fn add_totals_of_in_out_group(feature_column: &str, grouped_lf: &LazyFrame) -> LazyFrame {
        let in_group_lf = grouped_lf
            .clone()
            .group_by([col(feature_column)])
            .agg([col("count_in").sum().alias("total_in_group")]);

        let grouped_lf = grouped_lf.clone().join(
            in_group_lf,
            [col(feature_column)],
            [col(feature_column)],
            JoinArgs::new(JoinType::Left),
        );

        let out_group_lf =
            grouped_lf
                .clone()
                .group_by([col(feature_column)])
                .agg([col("count_out").sum().alias("total_out_group")]);

        let grouped_lf = grouped_lf.clone().join(
            out_group_lf,
            [col(feature_column)],
            [col(feature_column)],
            JoinArgs::new(JoinType::Left),
        );
        grouped_lf
    }

    fn group_in_and_out(lf: &mut LazyFrame, feature_column: &str) -> LazyFrame {
        let grouped_lf = lf
            .clone()
            .select([col(feature_column), col(TARGET_COLUMN)])
            .group_by([col(feature_column), col(TARGET_COLUMN)])
            .agg([col(TARGET_COLUMN)
                .count()
                .alias(COUNT_IN_COLUMN)
                .cast(DataType::Float64)])
            .with_columns([col(COUNT_IN_COLUMN)
                .sum()
                .over([col(TARGET_COLUMN)])
                .alias("total_per_target")])
            .with_columns([(col("total_per_target") - col(COUNT_IN_COLUMN)).alias("count_out")])
            .drop(["total_per_target"])
            .sort([feature_column, TARGET_COLUMN], Default::default());
        grouped_lf
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

        let (gini, sample_count) = compute_parent_gini_and_count_samples(&lf)?;
        assert_eq!(gini, 0.5941737597760909);
        assert_eq!(sample_count, 889.0);

        Ok(())
    }
}
