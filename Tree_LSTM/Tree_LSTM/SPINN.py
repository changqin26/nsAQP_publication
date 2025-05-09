
import torch
import torch.nn as nn
from Tree_LSTM.Tree_LSTM.TreeLSTM import TreeLSTM
# Note: no more need to import build_tree if you do it in the Dataset

class SPINN(nn.Module):
    def __init__(self, n_classes, size, embeddings_folder):
        super(SPINN, self).__init__()
        self.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        self.size = size

        self.tree_lstm = TreeLSTM(size)
        self.leaf_transform_h = nn.Linear(size, size)
        self.leaf_transform_c = nn.Linear(size, size)
        self.FC = nn.Linear(606, 303)
        self.mlp_classfication = nn.Sequential(
            nn.Linear(303, 303),
            nn.ReLU(),
            nn.Linear(303, n_classes)
        )
        self.sigmoid = nn.Sigmoid()

        self.tree_lstm_memory = {}
        self.node_info_dict = {}
        self.leaf_count = 0

    def forward(self, list_of_root_nodes):
        """
        Now we get a list of *pre-built* root_nodes from the DataLoader
        """
        all_hidden_states = []
        for root_node in list_of_root_nodes:
            # Clear caches for each tree
            self.node_info_dict.clear()
            self.tree_lstm_memory.clear()
            self.leaf_count = 0

            if root_node is None:
                # No tree? Just zero
                h = torch.zeros(self.size, device=self.device)
            else:
                h, c = self.process_tree(root_node)

            if h.dim() == 1:
                h = h.unsqueeze(0)

            all_hidden_states.append(h)

        batched_h = torch.cat(all_hidden_states, dim=0).to(self.device)
        mlp_output = self.mlp_classfication(batched_h)
        output = self.sigmoid(mlp_output)
        return output

    def process_tree(self, node):
        if node is None:
            return (
                torch.zeros(self.size, device=self.device),
                torch.zeros(self.size, device=self.device),
            )
        # Leaf node
        if node.left is None and node.right is None:
            if node.node_id not in self.tree_lstm_memory:
                h_leaf, c_leaf = self.leaf(node.feature_vector)
                self.tree_lstm_memory[node.node_id] = (h_leaf, c_leaf)
                self.node_info_dict[node.node_id] = node.feature_vector
            return self.tree_lstm_memory[node.node_id]

        # Internal node
        left_h, left_c = self.process_tree(node.left) if node.left else (torch.zeros(self.size), torch.zeros(self.size))
        right_h, right_c = self.process_tree(node.right) if node.right else (torch.zeros(self.size), torch.zeros(self.size))

        # inputX is basically FC of child feature vectors
        inputX = self.inputX(node) if (node.left and node.right) else torch.tensor(node.feature_vector, dtype=torch.float32, device=self.device)
        result = self.childrenNode(left_h, left_c, right_h, right_c, inputX)
        self.tree_lstm_memory[node.node_id] = result
        return result

    def leaf(self, feature_vector):
        self.leaf_count += 1
        leaf_tensor = torch.tensor(feature_vector, dtype=torch.float32, device=self.device).unsqueeze(0)
        raw_h = self.leaf_transform_h(leaf_tensor)
        raw_c = self.leaf_transform_c(leaf_tensor)
        h_leaf = torch.tanh(raw_h)
        c_leaf = torch.tanh(raw_c)
        return h_leaf, c_leaf

    def inputX(self, parent_node):
        left_ft = torch.tensor(parent_node.left.feature_vector, dtype=torch.float32, device=self.device)
        right_ft = torch.tensor(parent_node.right.feature_vector, dtype=torch.float32, device=self.device)
        concat = torch.cat([left_ft, right_ft], dim=0).unsqueeze(0)
        transformed = self.FC(concat)
        # Save this new vector for reference if you want
        new_vector = transformed.squeeze(0).tolist()
        self.node_info_dict[parent_node.node_id] = new_vector
        return transformed.squeeze(0)

    def childrenNode(self, left_h, left_c, right_h, right_c, inputX):
        return self.tree_lstm(inputX, (left_h, left_c), (right_h, right_c))
