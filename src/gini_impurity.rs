use crate::constants::TARGET_COLUMN;
use polars::prelude::{col, lit, JoinArgs, JoinType};
use polars_core::datatypes::DataType;
use polars_core::frame::DataFrame;
use polars_core::prelude::NamedFrom;
use polars_core::series::Series;
use polars_lazy::frame::LazyFrame;

// TODO: Add fail safe to ensure TARGET_COLUMN doesn't already exist in dataframe

const COUNT_IN_COLUMN: &str = "count_in";
const COUNT_OUT_COLUMN: &str = "count_out";
const GINI_IMPURITY_IN_GROUP: &str = "gini_in";


fn compute_parent_gini_impurity(
    lf: &LazyFrame,
) -> Result<f64, Box<dyn std::error::Error>> {
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
    Ok((gini_impurity))
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
            ((col(GINI_IMPURITY_IN_GROUP) * col("total_in_group"))
                / (col("total_in_group") + col("total_out_group")))
                .alias(GINI_IMPURITY_IN_GROUP),
        );

        normalized_lf = normalized_lf.with_column(
            ((col("gini_out") * col("total_out_group"))
                / (col("total_in_group") + col("total_out_group")))
                .alias("gini_out"),
        );

        // Get total Gini Impurity of each possible split:
        normalized_lf =
            normalized_lf.with_column((col(GINI_IMPURITY_IN_GROUP) + col("gini_out")).alias("normalized_child_gini"));
        normalized_lf
    }

    fn compute_gini_per_feature(feature_column: &str, grouped_lf: &mut LazyFrame) -> LazyFrame {
        let gini_out_column = "gini_out";
        let mut gini_lf = grouped_lf.clone().with_column(
            ((col(COUNT_IN_COLUMN) / col("total_in_group")).pow(lit(2.0))).alias(GINI_IMPURITY_IN_GROUP),
        );

        gini_lf = gini_lf.with_column(
            ((col(COUNT_OUT_COLUMN) / col("total_out_group")).pow(lit(2.0))).alias(gini_out_column),
        );

        gini_lf = gini_lf
            .select([
                col(feature_column),
                col(GINI_IMPURITY_IN_GROUP),
                col(gini_out_column),
            ])
            .group_by([col(feature_column)])
            .agg([col(GINI_IMPURITY_IN_GROUP).sum(), col(gini_out_column).sum()])
            .with_columns([
                (lit(1.0) - col(GINI_IMPURITY_IN_GROUP)).alias(GINI_IMPURITY_IN_GROUP),
                (lit(1.0) - col(gini_out_column)).alias(gini_out_column),
            ]);
        gini_lf
    }

    fn add_totals_of_in_out_group(feature_column: &str, grouped_lf: &LazyFrame) -> LazyFrame {
        let in_group_lf = grouped_lf
            .clone()
            .group_by([col(feature_column)])
            .agg([col(COUNT_IN_COLUMN).sum().alias("total_in_group")]);

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
                .agg([col(COUNT_OUT_COLUMN).sum().alias("total_out_group")]);

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
            .with_columns([(col("total_per_target") - col(COUNT_IN_COLUMN)).alias(COUNT_OUT_COLUMN)])
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

        let gini  = compute_parent_gini_impurity(&lf)?;
        assert_eq!(gini, 0.5941737597760909);

        Ok(())
    }
}
