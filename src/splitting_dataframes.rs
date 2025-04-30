use crate::rework::gini_impurity_working_file::{
    get_best_column_to_split_on, FEATURE_COLUMN_NAME, SELECTION_COLUMN, SORT_TYPE_COL,
};
use crate::rework::sort_type::SortType;
use crate::settings::Settings;
use polars::prelude::{col, lit, Expr};
use polars_core::frame::DataFrame;
use polars_lazy::prelude::LazyFrame;
use std::error::Error;
use std::str::FromStr;

fn get_split_predicate(collected: DataFrame) -> Result<Expr, Box<dyn Error>> {
    // Extract all relevant values:
    let column_name = collected
        .column(FEATURE_COLUMN_NAME)?
        .str()?
        .get(0)
        .unwrap();
    let sort_type = SortType::from_str(collected.column(SORT_TYPE_COL)?.str()?.get(0).unwrap());
    let selection = collected.column(SELECTION_COLUMN)?.str()?.get(0).unwrap();

    // Create predicate:
    let predicate: Expr;
    match sort_type {
        SortType::Ordinal => {
            let threshold = f64::from_str(selection)?;
            predicate = col(column_name).lt(lit(threshold));
        }
        SortType::Categorical => {
            predicate = col(column_name).eq(lit(selection));
        }
    }
    Ok(predicate)
}

#[derive(Clone)]
struct TreeRemake {
    left_node: Option<Box<TreeRemake>>,
    right_node: Option<Box<TreeRemake>>,
    split_expression: Option<Expr>,
    depth: u8,
    settings: Settings,
}

impl TreeRemake {
    pub fn default() -> Self {
        Self {
            left_node: None,
            right_node: None,
            split_expression: None,
            depth: 0,
            settings: Settings::default(),
        }
    }

    pub fn fit(&mut self, lf: &LazyFrame) {
        // // Check depth:
        // if self.depth >= self.settings.get_max_depth(){
        //     return
        // }
        //
        // // todo: Add target column
        // let split_info = get_best_column_to_split_on(&lf)?.collect()?;
        // let split_expression = get_split_predicate(split_info)?;
        // self.split_expression = Some(split_expression);
        //
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::TARGET_COLUMN;

    use crate::test_utils::get_preprocessed_test_dataframe;
    use polars::prelude::not;

    #[test]
    fn test_split_left_right() -> Result<(), Box<dyn std::error::Error>> {
        let mut lf = get_preprocessed_test_dataframe();
        lf = lf.drop([TARGET_COLUMN]);
        let target_column = "Pclass";
        lf = lf.rename([target_column], [TARGET_COLUMN], true);

        let collected = get_best_column_to_split_on(&lf)?.collect()?;
        println!("{:?}", collected);
        let predicate = get_split_predicate(collected)?;
        let left_lf = lf.clone().filter(predicate.clone());
        let right_lf = lf.filter(not(predicate));
        println!("{:?}", left_lf.clone().collect()?);
        println!("{:?}", right_lf.collect()?);

        let collected = get_best_column_to_split_on(&left_lf)?.collect()?;
        println!("{:?}", collected);
        let predicate = get_split_predicate(collected)?;
        let new_left = left_lf.clone().filter(predicate.clone());
        let new_right = left_lf.filter(not(predicate));
        println!("{:?}", new_left.collect()?);
        println!("{:?}", new_right.collect()?);

        Ok(())
    }
}
