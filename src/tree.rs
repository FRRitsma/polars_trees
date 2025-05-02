use crate::display_tree::BinaryTree;
use crate::extract_values::get_most_common_label;
use crate::preprocessing::add_target_column;
use polars::prelude::{not, Expr};
use polars_lazy::frame::LazyFrame;
use std::error::Error;
use std::fmt;

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
            return Err(Box::from(
                "A final tree can not be parent to a new tree".to_string(),
            ));
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



    fn display(&self) -> String {
        if self.is_final() {
            return self.label.unwrap().to_string();
        }
        self.split_expression.clone().unwrap().to_string()
    }
}

impl BinaryTree for Tree {
    fn get_left(&self) -> Option<&Self> {
        self.left_node.as_deref()
    }

    fn get_right(&self) -> Option<&Self> {
        self.right_node.as_deref()
    }

    fn display_string(&self) -> String {
        self.display()
    }
}

#[cfg(test)]
mod tests {
    use crate::display_tree::BinaryTree;
    use crate::test_utils::{get_raw_test_dataframe, TITANIC_TARGET_COLUMN};
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



}
