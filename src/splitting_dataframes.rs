use crate::constants::TARGET_COLUMN;
use crate::rework::constants::{
    FEATURE_COLUMN_NAME, SELECTION_COLUMN, SORT_TYPE_COL, TOTAL_LEFT_GROUP_COL,
    TOTAL_RIGHT_GROUP_COL,
};
use crate::rework::gini_impurity::get_best_column_to_split_on;
use crate::rework::sort_type::SortType;
use crate::settings::Settings;
use polars::prelude::{col, lit, not, Expr, UnionArgs};
use polars_core::frame::DataFrame;
use polars_core::prelude::{AnyValue, SortMultipleOptions};
use polars_lazy::dsl::concat;
use polars_lazy::prelude::LazyFrame;
use std::error::Error;
use std::str::FromStr;

const PREDICTED_LABEL_COL: &str = "PREDICTED_LABEL";
const INDEX_COL: &str = "INDEX";

fn get_size_left_size_right(collected: &DataFrame) -> Result<(u128, u128), Box<dyn Error>> {
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

    pub fn fit(&mut self, lf: &LazyFrame, target_column: &str) -> Result<(), Box<dyn Error>> {
        // Pre-processing step: Renaming provided target column to hardcoded target column.
        let lf = lf.clone().rename([target_column], [TARGET_COLUMN], true);
        self.private_fit(lf)?;
        Ok(())
    }

    fn private_fit(&mut self, lf: LazyFrame) -> Result<(), Box<dyn Error>> {
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
        let split_df = get_best_column_to_split_on(&lf).unwrap().collect().unwrap();
        let (sample_size_left, sample_size_right) = get_size_left_size_right(&split_df)?;
        let predicate = get_split_predicate(split_df).unwrap();
        self.split_expression = Some(predicate.clone());

        // Step 4: Fit children, or set as final:
        if sample_size_left < self.settings.get_min_leave_size() {
            self.left_node.as_deref_mut().unwrap().is_final = true;
        } else {
            let left_lf = lf.clone().filter(predicate.clone());
            self.left_node
                .as_deref_mut()
                .unwrap()
                .private_fit(left_lf)?;
        }
        if sample_size_right < self.settings.get_min_leave_size() {
            self.right_node.as_deref_mut().unwrap().is_final = true;
        } else {
            let right_lf = lf.filter(not(predicate));
            self.right_node
                .as_deref_mut()
                .unwrap()
                .private_fit(right_lf)?;
        }

        Ok(())
    }

    pub fn predict(&self, lf: &LazyFrame) -> LazyFrame {
        let mut prediction_lf = lf.clone();
        // Add columns for prediction and index:
        let prediction_lf = prediction_lf
            .with_column(lit("").alias(PREDICTED_LABEL_COL))
            .with_row_index(INDEX_COL, None);
        return self.private_predict(prediction_lf);
    }

    fn private_predict(&self, lf: LazyFrame) -> LazyFrame {
        // If self is final, add label and return:
        if self.is_final {
            let label = self.label.clone().unwrap().clone();
            return lf
                .clone()
                .with_column(lit(label).alias(PREDICTED_LABEL_COL));
        }

        // If not final, send to child nodes:
        let mut left_lf = lf.clone().filter(self.split_expression.clone().unwrap());
        let mut right_lf = lf
            .clone()
            .filter(not(self.split_expression.clone().unwrap()));

        // Get predictions:
        if let (Some(left), Some(right)) = (&self.left_node, &self.right_node) {
            left_lf = left.private_predict(left_lf);
            right_lf = right.private_predict(right_lf);
        }

        // Combine:
        concat(vec![left_lf, right_lf], UnionArgs::default()).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::TARGET_COLUMN;

    use crate::rework::constants::{TOTAL_LEFT_GROUP_COL, TOTAL_RIGHT_GROUP_COL};
    use crate::test_utils::get_preprocessed_test_dataframe;
    use polars::prelude::not;
    use polars_core::prelude::SortMultipleOptions;
    use polars_core::utils::Container;

    #[test]
    fn test_split_left_right() -> Result<(), Box<dyn Error>> {
        let mut lf = get_preprocessed_test_dataframe();
        lf = lf.drop([TARGET_COLUMN]);
        let target_column = "Pclass";
        lf = lf.rename([target_column], [TARGET_COLUMN], true);
        let collected = get_best_column_to_split_on(&lf)?.collect()?;
        let (size_left, size_right) = get_size_left_size_right(&collected)?;
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
        tree.fit(&lf, target_column)?;

        // Testing expected outputs:
        assert!(tree.is_final);
        assert_eq!(tree.label.unwrap(), "3");
        Ok(())
    }

    #[test]
    fn test_fit_tree_with_depth_1() -> Result<(), Box<dyn Error>> {
        // Get lazyframe:
        let mut lf = get_preprocessed_test_dataframe();
        lf = lf.drop([TARGET_COLUMN]);
        let target_column = "Pclass";

        // Get tree with depth zero:
        let mut tree = ClassificationTree::default();
        tree.settings.set_max_depth(1);

        // Fit tree:
        tree.fit(&lf, target_column)?;

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

    // #[test]
    // fn test_depth_two_private_fit() -> Result<(), Box<dyn Error>> {
    //     let mut lf = get_preprocessed_test_dataframe();
    //     lf = lf.drop([TARGET_COLUMN]);
    //     let target_column = "Pclass";
    //     lf = lf.rename([target_column], [TARGET_COLUMN], true);
    //
    //     let mut tree = ClassificationTree::default();
    //     tree.settings.set_max_depth(2);
    //
    //     tree.private_fit(&lf)?;
    //
    //     println!(
    //         "{}",
    //         tree.left_node
    //             .clone()
    //             .unwrap()
    //             .left_node
    //             .unwrap()
    //             .label
    //             .unwrap()
    //     );
    //     println!(
    //         "{}",
    //         tree.left_node.unwrap().right_node.unwrap().label.unwrap()
    //     );
    //     Ok(())
    // }
    //
    // #[test]
    // fn test_predict() {
    //     let mut lf = get_preprocessed_test_dataframe();
    //     lf = lf.drop([TARGET_COLUMN]);
    //     let target_column = "Pclass";
    //     lf = lf.rename([target_column], [TARGET_COLUMN], true);
    //
    //     println!("{:?}", lf.collect());
    //
    //     // let mut tree = ClassificationTree::default();
    //     // tree.settings.set_max_depth(1);
    //     //
    //     // tree.private_fit(&lf).unwrap();
    //     //
    //     // let lf_predict = tree.predict(&lf);
    //     // assert!(tree.left_node.unwrap().is_final);
    //     // assert!(tree.right_node.unwrap().is_final);
    //     // println!("{:?}", lf_predict.collect());
    // }
}
