use std::cmp::Ordering;
use std::fmt;

pub trait BinaryTree {
    fn get_left(&self) -> Option<&Self>;
    fn get_right(&self) -> Option<&Self>;
    fn display_string(&self) -> String;
    fn display_tree(&self) -> DisplayTree {
        DisplayTree::fit_display_tree(self)
    }
}

#[derive(Clone)]
pub struct DisplayTree {
    left_node: Option<Box<DisplayTree>>,
    right_node: Option<Box<DisplayTree>>,
    x_position: Option<f32>,
    depth: u8,
    column_width: Option<usize>,
    display_offset: Option<usize>,
    expression: String,
}

impl PartialOrd for DisplayTree {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for DisplayTree {
    fn cmp(&self, other: &Self) -> Ordering {
        // First compare y values
        match self.depth.partial_cmp(&other.depth) {
            Some(Ordering::Equal) => {
                // If y is equal, compare x values
                self.x_position
                    .partial_cmp(&other.x_position)
                    .unwrap_or(Ordering::Equal)
            }
            Some(ordering) => ordering,
            None => Ordering::Equal, // handle NaN cases if needed
        }
    }
}

impl PartialEq for DisplayTree {
    fn eq(&self, other: &Self) -> bool {
        self.depth == other.depth && self.x_position == other.x_position
    }
}

impl Eq for DisplayTree {}

impl DisplayTree {
    fn new(expression: String, depth: u8) -> Self {
        Self {
            left_node: None,
            right_node: None,
            x_position: None,
            column_width: None,
            display_offset: None,
            depth,
            expression,
        }
    }

    pub fn fit_display_tree<T: BinaryTree + ?Sized>(input: &T) -> Self {
        let mut tree = Self::_fit_display_tree_recursive(input, 0);
        tree.format_tree();
        tree
    }

    fn _fit_display_tree_recursive<T: BinaryTree + ?Sized>(input: &T, depth: u8) -> Self {
        let mut tree = Self::new(input.display_string(), depth);
        if let Some(left) = input.get_left() {
            tree.left_node = Some(Box::new(Self::_fit_display_tree_recursive(left, depth + 1)));
        }
        if let Some(right) = input.get_right() {
            tree.right_node = Some(Box::new(Self::_fit_display_tree_recursive(
                right,
                depth + 1,
            )));
        }
        tree
    }

    fn padding(&self) -> usize {
        self.column_width
            .unwrap()
            .saturating_sub(self.display_offset.unwrap() + self.raw_display_len())
    }

    fn mid_point(&self) -> usize{
        self.display_offset.unwrap() + (self.raw_display_len() / 2)
    }

    fn format_tree(&mut self) {
        self.add_missing_nodes();
        self.assign_horizontal_order();
        self.set_column_width();
        self.assign_offset();
    }

    fn get_max_depth(&self) -> u8 {
        match (&self.left_node, &self.right_node) {
            (Some(left), Some(right)) => left.get_max_depth().max(right.get_max_depth()),
            (Some(left), None) => left.get_max_depth(),
            (None, Some(right)) => right.get_max_depth(),
            (None, None) => self.depth,
        }
    }

    fn add_missing_nodes(&mut self) {
        let max_depth = self.get_max_depth();
        self._add_missing_nodes(max_depth);
    }

    fn _add_missing_nodes(&mut self, max_depth: u8) {
        // Exit if current node is at max depth:
        if self.depth >= max_depth {
            return;
        }
        match (&mut self.left_node, &mut self.right_node) {
            // Check child nodes:
            (Some(left), Some(right)) => {
                left._add_missing_nodes(max_depth);
                right._add_missing_nodes(max_depth);
            }
            (Some(left), None) => {
                left._add_missing_nodes(max_depth);
            }
            (None, Some(right)) => {
                right._add_missing_nodes(max_depth);
            }
            // Create "empty" node for printing consistency:
            (None, None) => {
                self.left_node = Some(Box::from(DisplayTree::new("".to_string(), self.depth + 1)));
                self.left_node
                    .as_deref_mut()
                    .unwrap()
                    ._add_missing_nodes(max_depth);
            }
        }
    }

    pub fn set_column_width(&mut self) {
        _ = self._set_column_width(0);
    }
    fn _set_column_width(&mut self, parent_width: usize) -> usize {
        let self_width = self.raw_display_len().max(parent_width);

        let child_width: usize;

        match (&mut self.left_node, &mut self.right_node) {
            (Some(left), Some(right)) => {
                let left_width = left._set_column_width(0);
                let right_width = right._set_column_width(0);

                if (left_width + right_width) < self_width {
                    child_width = left._set_column_width(self_width / 2)
                        + right._set_column_width(self_width / 2);
                } else {
                    child_width = left_width + right_width;
                }
            }
            (Some(left), None) => {
                child_width = left._set_column_width(self_width);
            }
            (None, Some(right)) => {
                child_width = right._set_column_width(self_width);
            }
            (None, None) => {
                child_width = 0;
            }
        }

        let max_width = self_width.max(child_width);

        self.column_width = Some(max_width);

        return max_width;
    }

    pub fn assign_horizontal_order(&mut self) {
        self._assign_horizontal_order(0.5, 0.0);
    }

    fn _assign_horizontal_order(&mut self, scale: f32, horizontal_order: f32) {
        self.x_position = Some(horizontal_order);
        match (&mut self.left_node, &mut self.right_node) {
            (Some(left), Some(right)) => {
                let new_scale = scale / 2.0;
                left._assign_horizontal_order(new_scale, horizontal_order - scale);
                right._assign_horizontal_order(new_scale, horizontal_order + scale);
            }
            (Some(left), None) => {
                left._assign_horizontal_order(scale, horizontal_order);
            }
            (None, Some(right)) => {
                right._assign_horizontal_order(scale, horizontal_order);
            }
            (None, None) => {
                // Leaf node, nothing to do
            }
        }
    }

    fn assign_offset(&mut self) {
        let total_empty_length = self
            .column_width
            .expect("'column_width' was accessed before it was computed")
            .saturating_sub(self.raw_display_len());

        let raw_len = self.raw_display_len();
        let maximum_offset = self.column_width.unwrap() - raw_len;
        let ideal_offset: usize;

        match (&mut self.left_node, &mut self.right_node) {
            (Some(left), Some(right)) => {
                ideal_offset = left.column_width.unwrap() - (raw_len / 2);
                left.assign_offset();
                right.assign_offset();
            }
            (Some(left), None) => {
                ideal_offset = total_empty_length / 2;
                left.assign_offset();
            }
            (None, Some(right)) => {
                ideal_offset = total_empty_length / 2;
                right.assign_offset();
            }
            (None, None) => {
                ideal_offset = total_empty_length / 2;
            }
        }
        self.display_offset = Some(maximum_offset.min(ideal_offset));
    }

    // Display logic:
    pub fn raw_display(&self) -> String {
        format!(" {}", self.expression)
    }

    pub fn raw_display_len(&self) -> usize {
        self.raw_display().len()
    }

    pub fn padded_display(&self) -> String {
        let padding_left_string = " ".repeat(self.display_offset.unwrap());
        let padding_right_string = " ".repeat(self.padding());
        format!(
            "{}{}{}",
            padding_left_string,
            self.raw_display(),
            padding_right_string
        )
    }

    // Collecting nodes:
    pub fn all_nodes(&self) -> Vec<&DisplayTree> {
        let mut result = Vec::new();
        self.collect_nodes(&mut result);
        result
    }

    fn collect_nodes<'a>(&'a self, vec: &mut Vec<&'a DisplayTree>) {
        vec.push(self);
        if let Some(ref left) = self.left_node {
            left.collect_nodes(vec);
        }
        if let Some(ref right) = self.right_node {
            right.collect_nodes(vec);
        }
    }
}

impl fmt::Display for DisplayTree {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut nodes = self.all_nodes();
        nodes.sort();

        if let Some(first) = nodes.get(0) {
            write!(f, "{}", first.padded_display())?;
        }

        for chunk in nodes.windows(2) {
            if let [first, second] = chunk {
                if first.depth != second.depth {
                    writeln!(f)?;
                }
                write!(f, "{}", second.padded_display())?;
            }
        }

        writeln!(f)?; // Final newline
        Ok(())
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::empty_tree::EmptyTree;

    #[test]
    pub fn test_create_a_test_tree() {
        let tree = EmptyTree::new("Hello".to_string());
        println!("{}", tree.display_string());
    }

    #[test]
    pub fn fit_a_display_tree_to_a_test_tree_depth_zero() {
        let tree = EmptyTree::new("Hello".to_string());
        println!("{}", tree.display_tree());
    }

    #[test]
    pub fn fit_a_display_tree_to_a_test_tree_depth_one() {
        let mut tree = EmptyTree::new("Hello".to_string());
        tree.left_node = Some(Box::from(EmptyTree::new("World".to_string())));
        tree.right_node = Some(Box::from(EmptyTree::new("Goodbye".to_string())));
        println!("{}", tree.display_tree());
    }

    #[test]
    pub fn display_a_very_long_string() {
        let mut tree = EmptyTree::new("AVERYLONGSTRINGAVERYLONGSTRINGAVERYLONGSTRING".to_string());
        tree.left_node = Some(Box::from(EmptyTree::new("!".to_string())));
        tree.right_node = Some(Box::from(EmptyTree::new("?".to_string())));
        tree.right_node.as_deref_mut().unwrap().left_node =
            Some(Box::from(EmptyTree::new("?".to_string())));
        tree.right_node.as_deref_mut().unwrap().right_node =
            Some(Box::from(EmptyTree::new("!".to_string())));
        println!("{}", tree.display_tree());
    }

    #[test]
    pub fn display_the_first_node_shifted_to_the_right() {
        let mut tree = EmptyTree::new("nnnnnn".to_string());
        tree.left_node = Some(Box::from(EmptyTree::new(
            "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!".to_string(),
        )));
        tree.right_node = Some(Box::from(EmptyTree::new("??".to_string())));
        println!("{}", tree.display_tree());
    }
}
