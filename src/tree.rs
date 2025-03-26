use crate::extract_values::get_most_common_label;
use crate::generic_functions::get_optimal_leaf_value_of_dataframe;
use crate::preprocessing::add_target_column;
use polars::prelude::{not, Expr};
use polars_lazy::frame::LazyFrame;
use std::error::Error;

struct Tree {
    left_node: Option<Box<Tree>>,
    right_node: Option<Box<Tree>>,
    split_expression: Option<Expr>,
    end_node: bool,
    label: Option<bool>,
    depth: u8,
    minimal_sample_size: u16,
    max_depth: u8,
}

impl Tree {
    pub fn new(minimal_sample_size: u16, max_depth: u8) -> Self {
        Self {
            left_node: None,
            right_node: None,
            split_expression: None,
            end_node: false,
            label: None,
            depth: 0,
            minimal_sample_size,
            max_depth,
        }
    }

    fn private_new(&self) -> Result<Self, Box<dyn Error>> {
        if self.is_final() {
            return Err(
                Box::try_from("A final tree can not be parent to a new tree".to_string()).unwrap(),
            );
        }
        Ok(Self {
            left_node: None,
            right_node: None,
            split_expression: None,
            end_node: false,
            label: None,
            depth: self.depth + 1,
            minimal_sample_size: self.minimal_sample_size,
            max_depth: self.max_depth,
        })
    }

    pub fn is_final(&self) -> bool {
        return self.depth >= self.max_depth || self.end_node;
    }

    pub fn fit(&mut self, df: LazyFrame, target_column: &str) {
        // Public fit defers to private fit to ensure add_target_column is performed only once
        let df = add_target_column(df, target_column);
        self._private_fit(df);
    }

    fn _private_fit(&mut self, df: LazyFrame) {
        if self.is_final() {
            let label = get_most_common_label(&df).unwrap();
            self.label = Some(label);
            return;
        }

        // Get split expression:
        let leaf_value = get_optimal_leaf_value_of_dataframe(&df).unwrap();
        let split_expression = leaf_value.split_expression;

        // Define left and right node:
        let left_node = self.private_new().unwrap();
        let right_node = self.private_new().unwrap();

        // Define left and right dataframe:
        let left = df.clone().filter(split_expression.clone());
        let right = df.filter(not(split_expression));

        // Fit one child nodes:

        // Assign to self:

    }
}

#[cfg(test)]
mod tests {
    use crate::test_utils::{get_test_dataframe, TITANIC_TARGET_COLUMN};
    use crate::tree::Tree;
    use polars_lazy::prelude::IntoLazy;

    #[test]
    pub fn test_zero_depth_is_final() {
        let tree: Tree = Tree::new(100, 0);
        assert_eq!(tree.is_final(), true);
    }

    #[test]
    pub fn test_one_depth_is_not_final() {
        let tree: Tree = Tree::new(100, 1);
        assert_eq!(tree.is_final(), false);
    }

    #[test]
    pub fn test_private_new_increases_depth() {
        let tree: Tree = Tree::new(100, 1);
        assert_eq!(tree.depth, 0);
        let new_tree = tree.private_new().unwrap();
        assert_eq!(new_tree.depth, 1);
    }

    #[test]
    pub fn test_zero_depth_tree_labels_false() {
        let df = get_test_dataframe().lazy();
        let mut tree: Tree = Tree::new(100, 0);
        tree.fit(df, TITANIC_TARGET_COLUMN);
        assert_eq!(tree.label, Some(false));
    }
}
