import torch
from pathlib import Path
import yaml
import torch.nn as nn
import torch_geometric


from dataclasses import asdict, dataclass


@dataclass(eq=False, repr=False)
class GNNPlanner(nn.Module):
    obs_radius: int = 5
    dim_hidden: int = 128
    num_encoder_layers: int = 2
    num_gnn_layers: int = 3
    num_decoder_layers: int = 2
    heads: int = 1

    # should be aligned with observations
    num_channels: int = 2
    edge_dim: int = 3
    num_classes: int = 5

    def __post_init__(self) -> None:
        super().__init__()

        # encoder
        dim_first_layer = self.num_channels * (self.obs_radius * 2 + 1) ** 2
        layers = [nn.Flatten(1)]
        for i in range(self.num_encoder_layers):
            layers.extend(
                [
                    nn.Linear(
                        self.dim_hidden if i > 0 else dim_first_layer, self.dim_hidden
                    ),
                    nn.ReLU(),
                ]
            )
        self.encoder = nn.Sequential(*layers)

        # GNN
        self.gnn = torch_geometric.nn.models.GAT(
            in_channels=self.dim_hidden,
            hidden_channels=self.dim_hidden // self.heads,
            num_layers=self.num_gnn_layers,
            out_channels=self.dim_hidden,
            v2=True,
            edge_dim=self.edge_dim,
            add_self_loops=False,
            heads=self.heads,
            concat=True,
        )

        # decoder
        layers = []
        for _ in range(self.num_decoder_layers):
            layers.extend(
                [
                    nn.Linear(self.dim_hidden, self.dim_hidden),
                    nn.ReLU(),
                ]
            )
        layers.append(nn.Linear(self.dim_hidden, self.num_classes))
        self.decoder = nn.Sequential(*layers)

    def forward(self, data, **_) -> torch.Tensor:
        x_enc = self.encoder(data["x"])
        x_comm = self.gnn(
            x_enc, edge_index=data["edge_index"], edge_attr=data["edge_attr"]
        )
        x = self.decoder(x_comm + x_enc)
        return x

    def save(self, filename: Path | str) -> None:
        torch.save(self.state_dict(), filename)
        with open(f"{filename}_hypra.yaml", "w") as f:
            yaml.dump(asdict(self), f)

    @staticmethod
    def reconstruct(fpath: Path | str, device: str = "cpu", **kwargs):
        with open(f"{fpath}_hypra.yaml", "r") as f:
            model = GNNPlanner(**{**yaml.safe_load(f), **kwargs})
        model.load_state_dict(torch.load(open(fpath, "rb"), map_location=device))
        return model
