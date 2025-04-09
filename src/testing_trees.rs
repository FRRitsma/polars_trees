use std::cmp::Ordering;
use std::fmt;

trait BinaryTree {
    fn get_left(&self) -> Option<&Self>;
    fn get_right(&self) -> Option<&Self>;
    fn display_string(&self) -> String;
}

struct TestTree {
    left_node: Option<Box<TestTree>>,
    right_node: Option<Box<TestTree>>,
    string: String
}

impl BinaryTree for TestTree{
    fn get_left(&self) -> Option<&Self> {
        self.left_node.as_deref()
    }

    fn get_right(&self) -> Option<&Self> {
        self.right_node.as_deref()
    }

    fn display_string(&self) -> String{
        self.string.to_string()
    }
}

impl TestTree {
    pub fn new(string: String) -> Self{
        Self{
            left_node: None,
            right_node: None,
            string
        }
    }
}



#[derive(Clone)]
struct DisplayTree {
    left_node: Option<Box<DisplayTree>>,
    right_node: Option<Box<DisplayTree>>,
    placement: FormatPlacement,
    x_position: f32,
    depth: u8,
    column_width: usize,
    expression: String,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum FormatPlacement {
    Left,
    Right,
    Center,
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

    pub fn fit_display_tree<T: BinaryTree>(input: &T) -> Self {
        let mut tree = Self::_fit_display_tree_recursive(input, 0);
        tree.assign_order();
        tree.set_column_width();
        tree
    }

    fn _fit_display_tree_recursive<T: BinaryTree>(input: &T, depth: u8) -> Self {
        let mut tree = Self::_new(input.display_string(), depth);
        if let Some(left) = input.get_left() {
            tree.left_node = Some(Box::new(Self::_fit_display_tree_recursive(left, depth + 1)));
        }
        if let Some(right) = input.get_right() {
            tree.right_node = Some(Box::new(Self::_fit_display_tree_recursive(right, depth + 1)));
        }
        tree
    }

    fn _new(expression: String, depth: u8) -> Self {
        Self {
            left_node: None,
            right_node: None,
            placement: FormatPlacement::Center,
            x_position: 0.0,
            column_width: 0,
            depth,
            expression,
        }
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
                self.add_left("".to_string());
                self.left_node
                    .as_deref_mut()
                    .unwrap()
                    ._add_missing_nodes(max_depth);
            }
        }
    }

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

    pub fn _assign_placement(&mut self, parent_placement: FormatPlacement) -> FormatPlacement {
        /*
        Rules for placement, in order:
        0. The root node is always centered
        1. If node has two children, center self
        2. If node has one child and it is centered, center self
        3. Assume parent_placement, and pass it on to child
        */


        let self_placement: FormatPlacement;

        match (&mut self.left_node, &mut self.right_node) {
            (Some(left), Some(right)) => {
                self_placement = FormatPlacement::Center;
                _ = left._assign_placement(FormatPlacement::Left);
                _ = right._assign_placement(FormatPlacement::Right);
            }
            (Some(left), None) => {
                let child_placement = left._assign_placement(FormatPlacement::Left);
                // Center self if child node is centered:
                if child_placement == FormatPlacement::Center {
                    self_placement = child_placement;
                }
                // If not, assign parent placement:
                else {
                    self_placement = parent_placement;
                }
            }
            (None, Some(right)) => {
                let child_placement = right._assign_placement(FormatPlacement::Right);
                // Center self if child node is centered:
                if child_placement == FormatPlacement::Center {
                    self_placement = child_placement;
                }
                // If not, assign parent placement:
                else {
                    self_placement = parent_placement;
                }
            }
            (None, None) => {
                self_placement = parent_placement;
            }
        }

        // Assign to self:
        self.placement = self_placement;
        // Communicate placement to parent:
        self_placement
    }

    pub fn set_column_width(&mut self) {
        _ = self._set_column_width(0);
    }
    fn _set_column_width(&mut self, parent_width: usize) -> usize {
        let self_width = self.raw_display_len().max(parent_width);
        let child_width: usize;
        match (&mut self.left_node, &mut self.right_node) {
            (Some(left), Some(right)) => {
                child_width = left._set_column_width(0) + right._set_column_width(0);
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
        self.column_width = max_width;
        return max_width;
    }

    pub fn assign_order(&mut self) {
        // Entry point of recursive function:
        self.add_missing_nodes();
        self._assign_order(0.5, 0.0);
        self._assign_placement(FormatPlacement::Center);
    }

    fn _assign_order(&mut self, scale: f32, horizontal_order: f32) {
        self.x_position = horizontal_order;
        match (&mut self.left_node, &mut self.right_node) {
            (Some(left), Some(right)) => {
                let new_scale = scale / 2.0;
                left._assign_order(new_scale, horizontal_order - scale);
                right._assign_order(new_scale, horizontal_order + scale);
            }
            (Some(left), None) => {
                left._assign_order(scale, horizontal_order);
            }
            (None, Some(right)) => {
                right._assign_order(scale, horizontal_order);
            }
            (None, None) => {
                // Leaf node, nothing to do
            }
        }
    }

    pub fn add_left(&mut self, expression: String) {
        let new_y_position = self.depth + 1;
        self.left_node = Some(Box::from(DisplayTree::_new(expression, new_y_position)));
    }

    pub fn add_right(&mut self, expression: String) {
        let new_y_position = self.depth + 1;
        self.right_node = Some(Box::from(DisplayTree::_new(expression, new_y_position)));
    }

    pub fn raw_display(&self) -> String {
        self.expression.clone()
    }

    pub fn raw_display_len(&self) -> usize {
        self.expression.clone().len()
    }

    pub fn padded_display(&self) -> String {
        let total_padding_length = self.column_width.saturating_sub(self.raw_display_len());
        let (padding_left, padding_right) = match self.placement {
            FormatPlacement::Left => (0, total_padding_length),
            FormatPlacement::Right => (total_padding_length, 0),
            FormatPlacement::Center => {
                let left = total_padding_length / 2;
                let right = total_padding_length - left;
                (left, right)
            }
        };
        let padding_left_string = "-".repeat(padding_left);
        let padding_right_string = "-".repeat(padding_right);
        format!("{}{}{}", padding_left_string, self.raw_display(), padding_right_string)
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
    use std::hint::assert_unchecked;
    use super::*;

    #[test]
    pub fn test_create_a_test_tree() {
        let tree = TestTree::new("Hello".to_string());
        println!("{}", tree.display_string());
    }

    #[test]
    pub fn fit_a_display_tree_to_a_test_tree_depth_zero(){
        let tree = TestTree::new("Hello".to_string());
        let display_tree = DisplayTree::fit_display_tree(&tree);
        println!("{}", display_tree);
    }

    #[test]
    pub fn fit_a_display_tree_to_a_test_tree_depth_one(){
        let mut tree = TestTree::new("Hello".to_string());
        tree.left_node = Some(Box::from(TestTree::new("World".to_string())));
        tree.right_node = Some(Box::from(TestTree::new("Goodbye".to_string())));
        let display_tree = DisplayTree::fit_display_tree(&tree);
        let nodes = display_tree.all_nodes();
        println!("{}", display_tree);
        assert_eq!(nodes.len(), 3);
    }

    // #[test]
    // pub fn test_add_nodes() {
    //     let mut tree = DisplayTree::new("Hello".to_string());
    //     tree.add_left("World".to_string());
    //     tree.add_right("!".to_string());
    //     tree.assign_order();
    //
    //     let mut nodes = tree.all_nodes();
    //     nodes.sort();
    //     for n in nodes.iter() {
    //         println!("{}", n.x_position);
    //         println!("{}", n.depth);
    //         println!("{}", n.padded_display());
    //     }
    // }
    //
    // #[test]
    // pub fn test_add_nodes_compute_column_width() {
    //     let mut tree = DisplayTree::new("Hello".to_string());
    //     tree.add_left("Worlllllllld".to_string());
    //     tree.add_right("!".to_string());
    //     tree.right_node
    //         .as_deref_mut()
    //         .unwrap()
    //         .add_right("yoooo".to_string());
    //     tree.set_column_width();
    //     println!("{}", tree.padded_display());
    //     print!("{}", tree.left_node.unwrap().padded_display());
    //     println!("{}", tree.right_node.unwrap().padded_display());
    // }
    //
    // #[test]
    // pub fn test_get_max_depth() {
    //     let mut tree = DisplayTree::new("Hello".to_string());
    //     assert_eq!(tree.get_max_depth(), 0u8);
    //     tree.add_left("World".to_string());
    //     tree.add_right("!".to_string());
    //     assert_eq!(tree.get_max_depth(), 1u8);
    //     tree.right_node
    //         .as_deref_mut()
    //         .unwrap()
    //         .add_right("Goodbye".to_string());
    //     assert_eq!(tree.get_max_depth(), 2u8);
    // }
    //
    // #[test]
    // pub fn test_add_missing_nodes() {
    //     let mut tree = DisplayTree::new("Hello".to_string());
    //     tree.add_left("World".to_string());
    //     tree.add_right("!".to_string());
    //     tree.right_node
    //         .as_deref_mut()
    //         .unwrap()
    //         .add_right("Goodbye".to_string());
    //     let len_before = tree.all_nodes().len();
    //     tree.add_missing_nodes();
    //     let len_after = tree.all_nodes().len();
    //     assert!(len_after > len_before);
    // }
    //
    // #[test]
    // pub fn test_print_loop() {
    //     // let mut tree = TestTree::default("Hello".to_string());
    //     // tree.add_left("World".to_string());
    //     // tree.add_right("yooo".to_string());
    //     // tree.assign_position();
    //
    //     let mut tree = DisplayTree::new("Hello".to_string());
    //     tree.add_left("Worlllllllld".to_string());
    //     tree.add_right("!".to_string());
    //     tree.right_node
    //         .as_deref_mut()
    //         .unwrap()
    //         .add_right("yoooo".to_string());
    //     tree.assign_order();
    //     tree.set_column_width();
    //
    //     let mut nodes = tree.all_nodes();
    //     nodes.sort();
    //
    //     let first = nodes.get(0).unwrap();
    //     print!("{}", first.padded_display());
    //     for chunk in nodes.windows(2) {
    //         match chunk {
    //             [first, second] => {
    //                 if first.depth != second.depth {
    //                     println!();
    //                 }
    //                 print!("{}", second.padded_display());
    //             }
    //             _ => (),
    //         }
    //     }
    //     println!();
    // }
}
