from __future__ import annotations

import bz2
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Set

import networkx as nx
from pydantic import BaseModel, Field, ConfigDict
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class HetionetNode(BaseModel):
    """
    Minimal Hetionet node schema. Hetionet nodes typically include:
    - id (string)
    - kind (string) e.g. 'Disease', 'Gene', 'Compound', 'Pathway'
    - name (string)
    There may be other fields; we preserve them optionally if configured.
    """
    model_config = ConfigDict(extra="allow")

    id: str
    kind: str
    name: str


class HetionetEdge(BaseModel):
    """
    Minimal Hetionet edge schema. Hetionet edges typically include:
    - source_id
    - target_id
    - kind (relation type)
    There may be other fields.
    """
    model_config = ConfigDict(extra="allow")

    source_id: str
    target_id: str
    kind: str


class HetionetJson(BaseModel):
    """
    Root schema for Hetionet JSON export.
    """
    model_config = ConfigDict(extra="allow")

    nodes: List[HetionetNode]
    edges: List[HetionetEdge]


class HetionetLoadSettings(BaseSettings):
    """
    Loader settings for Hetionet JSON or JSON.BZ2.

    Environment variables:
      HETIONET_LOAD_INPUT_PATH=/path/to/hetionet-v1.0.json.bz2
      HETIONET_LOAD_KEEP_NODE_FIELDS=id,kind,name,whatever
      HETIONET_LOAD_KEEP_EDGE_FIELDS=kind,whatever
      HETIONET_LOAD_STRICT=false
    """
    INPUT_PATH: Path = Path.home() / ".cache" / "hetionet" / \
        "hetionet-v1.0.json.bz2"

    # Keep minimal by default; you can expand via env.
    KEEP_NODE_FIELDS: Set[str] = Field(
        default_factory=lambda: {"id", "kind", "name"})
    KEEP_EDGE_FIELDS: Set[str] = Field(
        default_factory=lambda: {"kind", "source_id", "target_id"})

    # If True, validate strict HetionetJson (can be memory heavy since it materializes lists).
    # If False, do a lighter parse (still loads full JSON) but only extracts needed fields.
    STRICT: bool = False

    model_config = SettingsConfigDict(env_prefix="HETIONET_LOAD_")


class HetionetLoader:
    """
    Single responsibility:
      - Read Hetionet JSON/JSON.BZ2
      - Enforce allowed/kept fields
      - Construct nx.DiGraph with node attrs and edge attrs

    Output graph contracts (generic):
      - Nodes: G.nodes[node_id][...] contains at least "kind" and "name" if kept
      - Edges: G.edges[u, v][...] contains at least "relation" derived from edge.kind
    """

    def __init__(self, settings: HetionetLoadSettings):
        self.settings = settings

    def load(self) -> nx.DiGraph:
        data = self._read_json(self.settings.INPUT_PATH)

        if self.settings.STRICT:
            parsed = HetionetJson.model_validate(data)
            nodes_iter = parsed.nodes
            edges_iter = parsed.edges
        else:
            # Lightweight extraction without building full pydantic tree for all nodes/edges
            nodes_iter = (HetionetNode.model_validate(n)
                          for n in data.get("nodes", []))
            edges_iter = (HetionetEdge.model_validate(e)
                          for e in data.get("edges", []))

        G = nx.DiGraph()

        # Add nodes
        for node in nodes_iter:
            attrs = self._filter_dict(
                node.model_dump(), self.settings.KEEP_NODE_FIELDS)
            node_id = attrs.get("id") or node.id
            # remove "id" from attrs since node_id is key
            attrs.pop("id", None)
            G.add_node(node_id, **attrs)

        # Add edges
        for edge in edges_iter:
            raw = edge.model_dump()
            kept = self._filter_dict(raw, self.settings.KEEP_EDGE_FIELDS)

            source_id = kept.get("source_id") or edge.source_id
            target_id = kept.get("target_id") or edge.target_id
            relation = kept.get("kind") or edge.kind

            # Remove source/target/kind from attrs after mapping
            kept.pop("source_id", None)
            kept.pop("target_id", None)
            kept.pop("kind", None)

            # Use "relation" as the generic attribute name (not Hetionet-specific)
            kept["relation"] = relation

            G.add_edge(source_id, target_id, **kept)

        logger.info("Loaded graph: %d nodes, %d edges",
                    G.number_of_nodes(), G.number_of_edges())
        return G

    @staticmethod
    def _filter_dict(d: Dict[str, Any], keep: Set[str]) -> Dict[str, Any]:
        return {k: v for k, v in d.items() if k in keep}

    @staticmethod
    def _read_json(path: Path) -> Dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(f"Hetionet input not found: {path}")

        if path.suffix == ".bz2":
            with bz2.open(path, "rt") as f:
                return json.load(f)

        with open(path, "rt") as f:
            return json.load(f)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    settings = HetionetLoadSettings()
    loader = HetionetLoader(settings)
    G = loader.load()
    print(G)
