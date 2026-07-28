"""
Locomo Statistics Runner

Inspects a built memory store and verifies the consistency of cue indices and
linked memories. Useful as a quick sanity check on the contents of a Chroma
collection.
"""

from curses import meta
import logging
import os
import hydra
from datetime import datetime
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

# Memory system implementations

from agent_memory.client import MemoryClient
from run_agent_memory import run_agent_memory_experiment
from chromadb import PersistentClient

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="./conf", config_name="config")
def run(cfg: DictConfig):
    """
    Hydra entry point.

    Walks every entry of a Chroma collection and verifies that cue indices
    point to existing primary memories and that linked memories are
    reachable.

    Args:
        cfg: Hydra configuration object containing all experiment parameters.

    Raises:
        ValueError: If an unsupported memory technique type is specified.
    """
    # Open the Chroma client (configure via cfg.memory.persist_path)
    # cfg.memory.memory_store = "external_baseline-4.1-subset1-cueindex"
    cfg.memory.memory_store = "external_baseline-4.1-subset1-cueindex-debug"
    client = PersistentClient(path=cfg.memory.persist_path)
    collection = client.get_collection(cfg.memory.collection_name)

    # Pull every record matching the user filter
    where_clause = {"user_id": "Melanie_0"}
    outcome = collection.get(where=where_clause, include=["metadatas", "documents"])

    # outcome = collection.get(where={"index": "Caroline friendship gratitude"}, include=["metadatas", "documents"])
    # print(outcome)

    for pos, (doc, metadata) in enumerate(zip(outcome["documents"], outcome["metadatas"])):
        # Examine both the document body and metadata
        memory_text = doc
        if metadata["linked_memory"]:
            # cue index entry
            cue_index = memory_text

            if "cue_indices" in metadata and metadata["cue_indices"]:
                print(f"Error: Both linked_memory and cue_indices exist for index {cue_index}!")

            num_words = len(cue_index.split(" "))
            linked_memories = metadata['linked_memory'].split("||")
            for linked in linked_memories:
                linked_result = collection.get(where={"index": linked}, include=["metadatas", "documents"])["ids"]
                if not linked_result:
                    print(f"[Error] Linked Memory Not Found: {cue_index} -> {linked}!")
            # if len(linked_memories) > 2:
            #     print(f"{pos:04d}[cue] {memory_text} -> {metadata['linked_memory']}")
            # if num_words == 2:
            #     print(f"{pos:04d}[cue] {memory_text} -> {metadata['linked_memory']}")
            # print(f"{pos:04d}[cue] {memory_text} -> {metadata['linked_memory']}")
        elif metadata["cue_indices"]:
            cue_indices = metadata['cue_indices'].split("||")
            for cue in cue_indices:
                cue_result = collection.get(where={"index": cue}, include=["metadatas", "documents"])["ids"]
                if not cue_result:
                    print(f"Error. Cue index {cue} doesn't exist!")
            if not metadata['cue_indices']:
                print("Error. No cue indices found!")
            primay_index = memory_text
            # print("-"*60)
            # print(f"{pos:04d}[pri] {memory_text} -> {metadata['cue_indices']}")
            # print(f"{metadata['value']}")


if __name__ == "__main__":
    run()
