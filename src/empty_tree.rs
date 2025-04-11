use crate::display_tree::BinaryTree;

pub struct EmptyTree {
    pub left_node: Option<Box<EmptyTree>>,
    pub right_node: Option<Box<EmptyTree>>,
    string: String,
}

impl BinaryTree for EmptyTree {
    fn get_left(&self) -> Option<&Self> {
        self.left_node.as_deref()
    }

    fn get_right(&self) -> Option<&Self> {
        self.right_node.as_deref()
    }

    fn display_string(&self) -> String {
        self.string.to_string()
    }
}

impl EmptyTree {
    pub fn new(string: String) -> Self {
        Self {
            left_node: None,
            right_node: None,
            string,
        }
    }
}
