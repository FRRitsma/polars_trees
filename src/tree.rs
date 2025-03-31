use crate::extract_values::get_most_common_label;
use crate::generic_functions::get_optimal_leaf_value_of_dataframe;
use crate::preprocessing::add_target_column;
use polars::prelude::{not, Expr, col};
use polars_lazy::frame::LazyFrame;
use std::error::Error;
use std::fmt;
use std::fmt::Formatter;
use std::cmp;


#[derive(Clone)]
struct Tree {
    left_node: Option<Box<Tree>>,
    right_node: Option<Box<Tree>>,
    split_expression: Option<Expr>,
    end_node: bool,
    label: Option<bool>,
    depth: u8,
    minimal_sample_size: f32,
    max_depth: u8,
}

impl fmt::Display for Tree {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {

        for n in 0..self.max_depth + 1 {
            writeln!(f, "{}", self.get_display_by_depth(n))?
        }
        Ok(())
    }
}

impl Tree {
    pub fn new(minimal_sample_size: f32, max_depth: u8) -> Self {
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
                Box::from("A final tree can not be parent to a new tree".to_string()),
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
        self.depth >= self.max_depth || self.end_node
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
        let leaf_value = get_optimal_leaf_value_of_dataframe(&df, self.minimal_sample_size).unwrap();
        let split_expression = leaf_value.split_expression;

        // Define left and right node:
        let mut left_node = self.private_new().unwrap();
        let mut right_node = self.private_new().unwrap();

        // Define left and right dataframe:
        let left = df.clone().filter(split_expression.clone());
        let right = df.filter(not(split_expression.clone()));

        left_node._private_fit(left);
        right_node._private_fit(right);

        self.left_node = Some(Box::from(left_node));
        self.right_node = Some(Box::from(right_node));
        self.split_expression = Some(split_expression);
    }

    fn display(&self) -> String {
        if self.is_final(){
            return self.label.unwrap().to_string();
        }
        return self.split_expression.clone().unwrap().to_string();
    }

    fn get_display_by_depth(&self, depth: u8) -> String {
        if self.depth == depth{
            return self.display();
        }
        else{
            let left_string = self.left_node.clone().unwrap().as_ref().get_display_by_depth(depth);
            let right_string = self.right_node.clone().unwrap().as_ref().get_display_by_depth(depth);
            let indent = "  ".repeat(depth.pow(2) as usize); // Repeats "  " depth times
            let total_string = left_string + &indent + &right_string;
            return total_string
        }

    }
}

#[cfg(test)]
mod tests {
    use crate::test_utils::{get_raw_test_dataframe, get_preprocessed_test_dataframe, TITANIC_TARGET_COLUMN};
    use crate::tree::Tree;
    use polars_lazy::prelude::IntoLazy;

    #[test]
    pub fn test_zero_depth_is_final() {
        let tree: Tree = Tree::new(100f32, 0);
        assert!(tree.is_final());
    }

    #[test]
    pub fn test_one_depth_is_not_final() {
        let tree: Tree = Tree::new(100f32, 1);
        assert!(!tree.is_final());
    }

    #[test]
    pub fn test_private_new_increases_depth() {
        let tree: Tree = Tree::new(100f32, 1);
        assert_eq!(tree.depth, 0);
        let new_tree = tree.private_new().unwrap();
        assert_eq!(new_tree.depth, 1);
    }

    #[test]
    pub fn test_zero_depth_tree_labels_false() {
        let df = get_raw_test_dataframe().lazy();
        let mut tree: Tree = Tree::new(100f32, 0);
        tree.fit(df, TITANIC_TARGET_COLUMN);
        assert_eq!(tree.label, Some(false));
    }

    #[test]
    pub fn test_depth_1() {
        let df = get_raw_test_dataframe().lazy();
        let mut tree: Tree = Tree::new(100f32, 1);
        tree.fit(df, TITANIC_TARGET_COLUMN);        // assert_eq!(tree.label, Some(false));
        println!("{}", tree.left_node.unwrap().label.unwrap());
        println!("{}", tree.right_node.unwrap().label.unwrap());
    }

    #[test]
    pub fn test_depth_2() {
        let df = get_raw_test_dataframe().lazy();
        let mut tree: Tree = Tree::new(100f32, 2);
        tree.fit(df, TITANIC_TARGET_COLUMN);
        println!("{}", tree);

    }

    #[test]
    pub fn test_depth_3() {
        let df = get_raw_test_dataframe().lazy();
        let mut tree: Tree = Tree::new(100f32, 3);
        tree.fit(df, TITANIC_TARGET_COLUMN);
        println!("{}", tree);
    }

    #[test]
    pub fn test_depth_4() {
        let df = get_raw_test_dataframe().lazy();
        let mut tree: Tree = Tree::new(100f32, 4);
        tree.fit(df, TITANIC_TARGET_COLUMN);
        println!("{}", tree);
    }

}
