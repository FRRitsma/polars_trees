use crate::constants::TARGET_COLUMN;
use crate::gini_impurity::constants::{
    FEATURE_COLUMN_NAME, SELECTION_COLUMN, SORT_TYPE_COL, TOTAL_LEFT_GROUP_COL,
    TOTAL_RIGHT_GROUP_COL,
};
use crate::gini_impurity::gini_impurity::get_gini_impurity_for_all_columns;
use crate::gini_impurity::sort_type::SortType;
use crate::old_preprocessing::pre_process_dataframe;
use crate::settings::Settings;
use polars::prelude::{col, lit, not, Expr, UnionArgs};
use polars_core::frame::DataFrame;
use polars_core::prelude::SortMultipleOptions;
use polars_lazy::dsl::concat;
use polars_lazy::prelude::LazyFrame;
use std::error::Error;
use std::str::FromStr;

const PREDICTED_LABEL_COL: &str = "PREDICTED_LABEL";
const INDEX_COL: &str = "INDEX";

fn get_size_of_left_and_right(collected: &DataFrame) -> Result<(u128, u128), Box<dyn Error>> {
    let size_left = collected
        .column(TOTAL_LEFT_GROUP_COL)?
        .f64()?
        .get(0)
        .unwrap() as u128;
    let size_right = collected
        .column(TOTAL_RIGHT_GROUP_COL)?
        .f64()?
        .get(0)
        .unwrap() as u128;
    Ok((size_left, size_right))
}

fn get_split_predicate(collected: DataFrame) -> Result<Expr, Box<dyn Error>> {
    // Extract all relevant values:
    let column_name = collected
        .column(FEATURE_COLUMN_NAME)?
        .str()?
        .get(0)
        .unwrap();
    let sort_type = SortType::from_str(collected.column(SORT_TYPE_COL)?.str()?.get(0).unwrap());
    let selection = collected.column(SELECTION_COLUMN)?.str()?.get(0).unwrap();

    if selection == ""{
        panic!("{} should not return an empty string", SELECTION_COLUMN);
    }

    // Create predicate:
    let predicate: Expr;
    match sort_type {
        SortType::Ordinal => {
            let threshold = f64::from_str(selection)?;
            predicate = col(column_name).gt(lit(threshold));
        }
        SortType::Categorical => {
            predicate = col(column_name).eq(lit(selection));
        }
    }
    Ok(predicate)
}

#[derive(Clone)]
struct ClassificationTree {
    // Generic tree properties:
    left_node: Option<Box<ClassificationTree>>,
    right_node: Option<Box<ClassificationTree>>,
    depth: u8,
    is_final: bool,

    // Polars specific:
    split_expression: Option<Expr>,
    label: Option<String>,

    // User defined settings:
    settings: Settings,
}

enum NodePosition {
    Left,
    Right,
}

fn get_most_common_label(lf: &LazyFrame) -> Result<String, Box<dyn Error>> {
    let mode_df = lf
        .clone()
        .group_by([TARGET_COLUMN])
        .agg([col(TARGET_COLUMN).count().alias("count")])
        .sort(
            ["count"],
            SortMultipleOptions::default().with_order_descending(true),
        )
        .collect()?;

    let mode = mode_df.column(TARGET_COLUMN)?.get(0).unwrap().to_string();
    Ok(mode)
}

impl ClassificationTree {
    pub fn default() -> Self {
        Self {
            left_node: None,
            right_node: None,
            split_expression: None,
            depth: 0,
            is_final: false,
            settings: Settings::default(),
            label: None,
        }
    }

    fn spawn_child(&mut self, node_position: NodePosition) {
        if self.settings.get_max_depth() < self.depth {
            panic!(
                "Max depth: {}. Current depth: {}",
                self.settings.get_max_depth(),
                self.depth
            );
        }

        let mut tree = ClassificationTree {
            left_node: None,
            right_node: None,
            split_expression: None,
            depth: self.depth + 1,
            is_final: false,
            settings: self.settings,
            label: None,
        };

        tree.is_final = tree.depth >= tree.settings.get_max_depth();

        match node_position {
            NodePosition::Left => {
                self.left_node = Some(Box::from(tree));
            }
            NodePosition::Right => {
                self.right_node = Some(Box::from(tree));
            }
        }
    }

    pub fn fit(&mut self, lf: LazyFrame, target_column: &str) -> Result<(), Box<dyn Error>> {
        // Pre-processing step: Renaming provided target column to hardcoded target column.
        let lf = pre_process_dataframe(lf, Settings::default(), target_column);
        self.private_fit(lf)?;
        Ok(())
    }

    fn private_fit(&mut self, lf: LazyFrame) -> Result<(), Box<dyn Error>> {
        let lf = lf.cache();

        // Step 1: Am I a final node?
        if self.depth == self.settings.get_max_depth() || self.is_final {
            self.is_final = true;
            self.label = Some(get_most_common_label(&lf).unwrap());
            return Ok(());
        }

        // Step 2: Create left/right node:
        self.spawn_child(NodePosition::Left);
        self.spawn_child(NodePosition::Right);

        // Step 3: Get the split criterion:
        let gini_lf = get_gini_impurity_for_all_columns(lf.clone())?.cache();
        let best_column = gini_lf.clone().first().collect()?;
        let (sample_size_left, sample_size_right) = get_size_of_left_and_right(&best_column)?;
        let predicate = get_split_predicate(best_column)?;
        self.split_expression = Some(predicate.clone());

        // // Clean columns:
        // let gini_df = gini_lf.collect()?;
        // let mut keep_columns = gini_df.column(FEATURE_COLUMN_NAME)?.str()?;
        // let mut keep_columns_vec = keep_columns
        //         .into_iter()
        //         .map(|opt_s| col(opt_s.unwrap_or(""))) // Provide default for None
        //         .collect::<Vec<_>>();
        // keep_columns_vec.push(col(TARGET_COLUMN));
        // let lf= lf.select(keep_columns_vec);

        // Step 4: Fit children
        // Step 4.a: Split lazyframe:
        let (left_lf, right_lf) = self.split_lazyframe_left_right(lf);

        // Step 4.b: Fit left
        let left_node = self.left_node.as_deref_mut().unwrap();
        if sample_size_left < self.settings.get_min_leave_size() {
            left_node.is_final = true;
        }
        left_node.private_fit(left_lf)?;

        // Step 4.c: Fit right
        let right_node = self.right_node.as_deref_mut().unwrap();
        if sample_size_right < self.settings.get_min_leave_size() {
            right_node.is_final = true;
        }
        right_node.private_fit(right_lf)?;

        Ok(())
    }

    pub fn predict(&self, lf: &LazyFrame) -> LazyFrame {
        let mut prediction_lf = lf.clone();
        // Add columns for prediction and index:
        prediction_lf = prediction_lf
            .with_column(lit("").alias(PREDICTED_LABEL_COL))
            .with_row_index(INDEX_COL, None);
        // Predict label, use index col to get back original ordering and then drop:
        let output_df = self
            .private_predict(prediction_lf)
            .sort([INDEX_COL], SortMultipleOptions::default())
            .drop([INDEX_COL]);
        output_df
    }

    fn private_predict(&self, lf: LazyFrame) -> LazyFrame {
        // If self is final, add label and return:
        if self.is_final {
            let label = self.label.clone().unwrap().clone();
            return lf.with_column(lit(label).alias(PREDICTED_LABEL_COL));
        }

        // If not final, send to child nodes:
        let (mut left_lf, mut right_lf) = self.split_lazyframe_left_right(lf);

        // Get predictions:
        if let (Some(left), Some(right)) = (&self.left_node, &self.right_node) {
            left_lf = left.private_predict(left_lf);
            right_lf = right.private_predict(right_lf);
        }

        // Combine:
        concat(vec![left_lf, right_lf], UnionArgs::default()).unwrap()
    }

    fn split_lazyframe_left_right(&self, lf: LazyFrame) -> (LazyFrame, LazyFrame) {
        let left_lf = lf.clone().filter(self.split_expression.clone().unwrap());
        let right_lf = lf.filter(not(self.split_expression.clone().unwrap()));
        (left_lf, right_lf)
    }
}

#[cfg(test)]
mod tests {
    use std::time::Instant;
    use super::*;
    use crate::constants::TARGET_COLUMN;

    use crate::test_utils::{get_preprocessed_test_dataframe, get_raw_test_dataframe};
    use polars::prelude::not;
    use polars_core::utils::Container;

    #[test]
    fn test_split_left_right() -> Result<(), Box<dyn Error>> {
        let mut lf = get_preprocessed_test_dataframe();
        lf = lf.drop([TARGET_COLUMN]);
        let target_column = "Pclass";
        lf = lf.rename([target_column], [TARGET_COLUMN], true);
        let collected = get_gini_impurity_for_all_columns(lf.clone())?.collect()?;
        let (size_left, size_right) = get_size_of_left_and_right(&collected)?;
        let predicate = get_split_predicate(collected)?;
        let left_lf = lf.clone().filter(predicate.clone()).collect()?;
        let right_lf = lf.filter(not(predicate)).collect()?;
        // Note: an error of size 1 was observed. Floating accuracy error?
        assert!(((right_lf.len() as i64) - (size_right as i64)).abs() < 2);
        assert!(((left_lf.len() as i64) - (size_left as i64)).abs() < 2);
        Ok(())
    }

    #[test]
    fn test_spawn_child_node() -> Result<(), Box<dyn Error>> {
        let mut tree = ClassificationTree::default();
        assert_eq!(tree.depth, 0);
        assert!(tree.settings.get_max_depth() > 0);
        tree.spawn_child(NodePosition::Right);
        assert!(tree.right_node.is_some());
        assert_eq!(tree.right_node.unwrap().depth, 1);
        Ok(())
    }

    #[test]
    fn test_fit_tree_with_depth_0() -> Result<(), Box<dyn Error>> {
        // Get lazyframe:
        let mut lf = get_preprocessed_test_dataframe();
        lf = lf.drop([TARGET_COLUMN]);
        let target_column = "Pclass";

        // Get tree with depth zero:
        let mut tree = ClassificationTree::default();
        tree.settings.set_max_depth(0);

        // Fit tree:
        tree.fit(lf.clone(), target_column)?;

        // Testing expected outputs:
        assert!(tree.is_final);
        assert_eq!(tree.label.unwrap(), "3");
        Ok(())
    }

    #[test]
    fn test_fit_tree_with_depth_1() -> Result<(), Box<dyn Error>> {
        // Get lazyframe:
        let mut lf = get_raw_test_dataframe();
        let target_column = "Pclass";

        // Get tree with depth one:
        let mut tree = ClassificationTree::default();
        tree.settings.set_max_depth(1);

        // Fit tree:
        tree.fit(lf.clone(), target_column)?;

        // Testing expected outputs:
        assert!(!tree.is_final);
        assert!(tree.left_node.as_ref().unwrap().is_final);
        assert!(tree.right_node.as_ref().unwrap().is_final);
        assert_eq!(tree.left_node.as_ref().unwrap().label.clone().unwrap(), "1");
        assert_eq!(
            tree.right_node.as_ref().unwrap().label.clone().unwrap(),
            "3"
        );

        Ok(())
    }

    #[test]
    fn test_predict_depth_0() -> Result<(), Box<dyn Error>> {
        // Get lazyframe:
        let mut lf = get_raw_test_dataframe();
        let target_column = "Pclass";

        // Get tree with depth zero:
        let mut tree = ClassificationTree::default();
        tree.settings.set_max_depth(0);

        // Fit tree:
        tree.fit(lf.clone(), target_column)?;
        let lf_predict = tree.predict(&lf);
        println!("{:?}", lf_predict.collect());
        Ok(())
    }

    #[test]
    fn test_predict_depth_1() -> Result<(), Box<dyn Error>> {
        // Get lazyframe:
        let mut lf = get_preprocessed_test_dataframe();
        lf = lf.drop([TARGET_COLUMN]);
        let target_column = "Pclass";

        // Get tree with depth zero:
        let mut tree = ClassificationTree::default();
        tree.settings.set_max_depth(1);

        // Fit tree:
        tree.fit(lf.clone(), target_column)?;
        let lf_predict = tree.predict(&lf);
        println!("{:?}", lf_predict.collect());
        Ok(())
    }

    #[test]
    fn test_predict_depth_2() -> Result<(), Box<dyn Error>> {
        // Get lazyframe:
        let lf = get_raw_test_dataframe();
        let target_column = "Pclass";

        // Get tree with depth zero:
        let mut tree = ClassificationTree::default();
        tree.settings.set_max_depth(2);

        // Fit tree:
        let start = Instant::now();
        tree.fit(lf.clone(), target_column)?;
        println!("Fitting took: {:?}", start.elapsed());

        let lf_predict = tree.predict(&lf);
        println!("{:?}", lf_predict.collect());
        Ok(())
    }

    #[test]
    fn test_predict_exceed_min_leave_size() -> Result<(), Box<dyn Error>> {
        // Get lazyframe:
        let mut lf = get_raw_test_dataframe();
        let target_column = "Pclass";

        // Get tree with depth zero:
        let mut tree = ClassificationTree::default();
        tree.settings.set_min_leave_size(400);

        // Fit tree:
        tree.fit(lf.clone(), target_column)?;
        let lf_predict = tree.predict(&lf);
        println!("{:?}", lf_predict.collect());
        Ok(())
    }
}
