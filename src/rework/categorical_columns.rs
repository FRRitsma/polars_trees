use crate::constants::TARGET_COLUMN;
use crate::preprocessing::REDUNDANT_STRING_VALUE;
use crate::rework::gini_impurity_working_file::{add_totals_of_in_out_group, add_zero_count, compute_gini_per_feature, extract_best_feature, normalize_gini_per_group, pre_process_for_gini};
use polars::prelude::{col, lit};
use polars_core::datatypes::DataType;
use polars_lazy::frame::LazyFrame;
use crate::rework::constants::{COUNT_LEFT_COL, COUNT_RIGHT_COL, SELECTION_COLUMN};
use crate::rework::gini_impurity_working_file;
use crate::rework::sort_type::SortType;

pub fn group_by_for_gini_impurity_categorical(lf: &LazyFrame) -> LazyFrame {
    // Instead of grouping by feature column, should group by selection column.
    let mut grouped_lf = lf
        // Group in and out:
        .clone()
        .group_by([col("*")])
        .agg([col(TARGET_COLUMN)
            .count()
            .alias(COUNT_LEFT_COL)
            .cast(DataType::Float64)]);

    grouped_lf = add_zero_count(SELECTION_COLUMN, TARGET_COLUMN, COUNT_LEFT_COL, &grouped_lf);

    grouped_lf = grouped_lf
        .with_columns([col(COUNT_LEFT_COL)
            .sum()
            .over([col(TARGET_COLUMN)])
            .alias("total_per_target")])
        .with_columns([(col("total_per_target") - col(COUNT_LEFT_COL)).alias(COUNT_RIGHT_COL)])
        .filter(col(SELECTION_COLUMN).neq(lit(REDUNDANT_STRING_VALUE)))
        .drop(["total_per_target"])
        .sort([SELECTION_COLUMN, TARGET_COLUMN], Default::default());
    grouped_lf
}

pub fn get_optimal_gini_impurity_for_categorical_column(
    lf: &LazyFrame,
    feature_column: &str,
) -> LazyFrame {
    let lf = pre_process_for_gini(&lf, SortType::Categorical, feature_column);
    let mut grouped_lf = group_by_for_gini_impurity_categorical(&lf);
    grouped_lf = add_totals_of_in_out_group(&grouped_lf);
    let gini_lf = compute_gini_per_feature(&grouped_lf);
    let normalized_gini_lf = normalize_gini_per_group(&grouped_lf, &gini_lf);
    let final_lf = extract_best_feature(normalized_gini_lf);
    final_lf
}


#[cfg(test)]
mod tests {
    use crate::constants::TARGET_COLUMN;
    use crate::test_utils::get_preprocessed_test_dataframe;
    use super::*;

    #[test]
    fn debug(){
        unsafe {
            std::env::set_var("POLARS_FMT_MAX_COLS", "100");
        }

        let mut lf = get_preprocessed_test_dataframe();
        lf = lf.drop([TARGET_COLUMN]);
        let feature_column = "Embarked";
        let target_column = "Pclass";
        lf = lf.rename([target_column], [TARGET_COLUMN], true);

        let lf = pre_process_for_gini(&lf, SortType::Categorical, feature_column);
        let mut grouped_lf = group_by_for_gini_impurity_categorical(&lf);
        grouped_lf = add_totals_of_in_out_group(&grouped_lf);
        let gini_lf = compute_gini_per_feature(&grouped_lf);
        let normalized_gini_lf = normalize_gini_per_group(&grouped_lf, &gini_lf);

        println!("{:?}", normalized_gini_lf.collect());
    }
}

