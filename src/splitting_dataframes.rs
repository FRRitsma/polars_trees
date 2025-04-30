use polars::prelude::Expr;
use polars_lazy::prelude::LazyFrame;
use crate::gini_impurity_working_file::get_best_column_to_split_on;
use crate::settings::Settings;

pub fn split_lazy_frame(lf: LazyFrame, settings: Settings, depth: u8){
    if depth == 0{
        return
    }

    // Check sample count:
    // get_best_column_to_split_on()

    // Should
    todo!()

}


#[derive(Clone)]
struct TreeRemake {
    left_node: Option<Box<TreeRemake>>,
    right_node: Option<Box<TreeRemake>>,
    split_expression: Option<Expr>,
    settings: Settings,
}

impl TreeRemake{
    pub fn default() -> Self {
        Self {
         left_node: None,
            right_node: None,
            split_expression: None,
            settings: Settings::default(),
        }
    }
}


#[cfg(test)]
mod tests {
    use std::error::Error;
    use std::str::FromStr;
    use polars::prelude::{col, Expr, lit, not};
    use polars_core::frame::DataFrame;
    use crate::constants::TARGET_COLUMN;
    use crate::gini_impurity_working_file::{FEATURE_COLUMN_NAME, SELECTION_COLUMN, SORT_TYPE, SortType};
    use crate::test_utils::get_preprocessed_test_dataframe;
    use super::*;

    #[test]
    fn test_split_left_right()-> Result<(), Box<dyn std::error::Error>>{
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

    fn get_split_predicate(collected: DataFrame) -> Result<Expr, Box<dyn Error>> {
        // Extract all relevant values:
        let column_name = collected.column(FEATURE_COLUMN_NAME)?.str()?.get(0).unwrap();
        let sort_type = SortType::from_str(collected.column(SORT_TYPE)?.str()?.get(0).unwrap());
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
}
