"""
Sanity Test for Memory Link Integrity

Tools for finding broken links between cues and linked memories inside an
agent_memory ChromaDB collection. Validates that every cue resolves to a
primary memory and that no orphaned references remain.
"""

import logging
import os
import hydra
from datetime import datetime
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from chromadb import PersistentClient
from typing import Dict, List, Set, Tuple

logger = logging.getLogger(__name__)


class MemoryLinkSanityTester:
    """
    Comprehensive integrity tester for memory links inside an agent_memory
    Chroma collection.

    Provides utilities to validate consistency between memory references,
    cue indices, and linked memories.
    """

    def __init__(self, client: PersistentClient, collection_name: str):
        """
        Construct a tester.

        Args:
            client: ChromaDB persistent client.
            collection_name: Name of the collection to test.
        """
        self.client = client
        self.collection_name = collection_name
        self.collection = client.get_collection(collection_name)

        # Counters
        self.stats = {
            'total_entries': 0,
            'cue_entries': 0,
            'primary_entries': 0,
            'broken_linked_memories': 0,
            'broken_cue_indices': 0,
            'orphaned_cues': 0,
            'orphaned_primaries': 0,
            'missing_cue_indices': 0,
            'missing_linked_memories': 0,
        }

        # Per-issue error logs
        self.errors = {
            'broken_linked_memories': [],
            'broken_cue_indices': [],
            'orphaned_cues': [],
            'orphaned_primaries': [],
            'missing_cue_indices': [],
            'missing_linked_memories': [],
        }

    def run_comprehensive_test(self, user_id: str = None) -> Dict:
        """
        Run the full integrity sweep against the collection.

        Args:
            user_id: Optional user filter (e.g. ``"Melanie_0"``).

        Returns:
            Dict containing test results and statistics.
        """
        print(f"🔍 Starting comprehensive sanity test for collection: {self.collection_name}")
        print(f"📊 User filter: {user_id if user_id else 'All users'}")
        print("-" * 80)

        # Where clause for user filtering
        where_clause = {"user_id": user_id} if user_id else None

        # Pull every entry from the collection
        outcome = self.collection.get(
            where=where_clause,
            include=["metadatas", "documents"]
        )

        self.stats['total_entries'] = len(outcome["documents"])
        print(f"📈 Total entries found: {self.stats['total_entries']}")

        # Run the individual integrity probes
        self._test_linked_memory_integrity(outcome)
        self._test_cue_index_integrity(outcome)
        self._test_orphaned_entries(outcome)

        return self._generate_report()

    def _test_linked_memory_integrity(self, outcome: Dict):
        """Walk every cue and confirm its linked memories exist."""
        print("\n🔗 Testing linked memory integrity...")

        for pos, (doc, metadata, entry_id) in enumerate(zip(
            outcome["documents"], outcome["metadatas"], outcome["ids"]
        )):
            memory_text = doc

            if metadata.get("linked_memory"):
                # Cue-index entry
                self.stats['cue_entries'] += 1
                cue_index = memory_text
                linked_memories = metadata['linked_memory'].split("||")

                for linked_memory in linked_memories:
                    linked_memory = linked_memory.strip()
                    if not linked_memory:
                        continue

                    # Verify the referenced memory exists
                    lookup = self.collection.get(
                        where={"index": linked_memory}
                    )

                    if not lookup["ids"]:
                        error_info = {
                            'cue_index': cue_index,
                            'missing_linked_memory': linked_memory,
                            'entry_id': entry_id,
                            'position': pos,
                        }
                        self.errors['broken_linked_memories'].append(error_info)
                        self.stats['broken_linked_memories'] += 1
                        print(f"❌ [Error] Linked Memory Not Found: {cue_index} -> {linked_memory}")
            else:
                # Possibly a primary memory entry
                if not metadata.get('cue_indices'):
                    self.stats['missing_linked_memories'] += 1
                    self.errors['missing_linked_memories'].append({
                        'entry_id': entry_id,
                        'memory_text': memory_text,
                        'position': pos,
                    })

    def _test_cue_index_integrity(self, outcome: Dict):
        """Verify that every primary memory's cue indices exist as entries."""
        print("\n🎯 Testing cue index integrity...")

        for pos, (doc, metadata, entry_id) in enumerate(zip(
            outcome["documents"], outcome["metadatas"], outcome["ids"]
        )):
            memory_text = doc

            if not metadata.get("linked_memory"):
                # Primary memory entry
                self.stats['primary_entries'] += 1

                if metadata.get('cue_indices'):
                    cue_indices = metadata['cue_indices'].split("||")

                    for cue_index in cue_indices:
                        cue_index = cue_index.strip()
                        if not cue_index:
                            continue

                        # Verify the cue index exists
                        cue_result = self.collection.get(
                            where={"index": cue_index}
                        )

                        if not cue_result["ids"]:
                            error_info = {
                                'primary_memory': memory_text,
                                'missing_cue_index': cue_index,
                                'entry_id': entry_id,
                                'position': pos,
                            }
                            self.errors['broken_cue_indices'].append(error_info)
                            self.stats['broken_cue_indices'] += 1
                            print(f"❌ [Error] Cue Index Not Found: {memory_text} -> {cue_index}")
                else:
                    # Primary memory with no cue indices at all
                    self.stats['missing_cue_indices'] += 1
                    self.errors['missing_cue_indices'].append({
                        'entry_id': entry_id,
                        'memory_text': memory_text,
                        'position': pos,
                    })
                    print(f"⚠️  [Warning] Primary memory without cue indices: {memory_text}")

    def _test_orphaned_entries(self, outcome: Dict):
        """Search for orphaned cue indices and primary memories."""
        print("\n🔍 Testing for orphaned entries...")

        # Sets of all known indices for fast lookup
        all_indices = set()
        linked_memories = set()
        cue_indices = set()

        for metadata in outcome["metadatas"]:
            if metadata.get("index"):
                all_indices.add(metadata["index"])

            if metadata.get("linked_memory"):
                for linked in metadata["linked_memory"].split("||"):
                    if linked.strip():
                        linked_memories.add(linked.strip())

            if metadata.get("cue_indices"):
                for cue in metadata["cue_indices"].split("||"):
                    if cue.strip():
                        cue_indices.add(cue.strip())

        # Orphaned cues: cues that don't point to any existing primary memory
        orphaned_cues = cue_indices - all_indices
        for orphaned_cue in orphaned_cues:
            self.stats['orphaned_cues'] += 1
            self.errors['orphaned_cues'].append(orphaned_cue)
            print(f"🔗 [Orphaned] Cue index references non-existent primary: {orphaned_cue}")

        # Orphaned primaries: linked memories that don't exist as indices
        orphaned_primaries = linked_memories - all_indices
        for orphaned_primary in orphaned_primaries:
            self.stats['orphaned_primaries'] += 1
            self.errors['orphaned_primaries'].append(orphaned_primary)
            print(f"💔 [Orphaned] Linked memory references non-existent index: {orphaned_primary}")

    def _generate_report(self) -> Dict:
        """Build and print the final test report."""
        print("\n" + "=" * 80)
        print("📋 SANITY TEST REPORT")
        print("=" * 80)

        # Summary stats
        print(f"📊 Total Entries: {self.stats['total_entries']}")
        print(f"🎯 Cue Entries: {self.stats['cue_entries']}")
        print(f"📝 Primary Entries: {self.stats['primary_entries']}")
        print()

        # Error counts
        print("❌ ERRORS FOUND:")
        print(f"   Broken Linked Memories: {self.stats['broken_linked_memories']}")
        print(f"   Broken Cue Indices: {self.stats['broken_cue_indices']}")
        print(f"   Orphaned Cues: {self.stats['orphaned_cues']}")
        print(f"   Orphaned Primaries: {self.stats['orphaned_primaries']}")
        print(f"   Missing Cue Indices: {self.stats['missing_cue_indices']}")
        print(f"   Missing Linked Memories: {self.stats['missing_linked_memories']}")
        print()

        # Health assessment
        total_errors = (
            self.stats['broken_linked_memories']
            + self.stats['broken_cue_indices']
            + self.stats['orphaned_cues']
            + self.stats['orphaned_primaries']
        )

        if total_errors == 0:
            print("✅ COLLECTION HEALTH: EXCELLENT - No broken links found!")
        elif total_errors < 10:
            print("⚠️  COLLECTION HEALTH: GOOD - Few issues found")
        elif total_errors < 50:
            print("🔧 COLLECTION HEALTH: NEEDS ATTENTION - Multiple issues found")
        else:
            print("🚨 COLLECTION HEALTH: CRITICAL - Many broken links found!")

        print("=" * 80)

        if total_errors == 0:
            health_status = 'excellent'
        elif total_errors < 10:
            health_status = 'good'
        elif total_errors < 50:
            health_status = 'attention'
        else:
            health_status = 'critical'

        return {
            'stats': self.stats,
            'errors': self.errors,
            'health_status': health_status,
        }

    def save_detailed_report(self, output_path: str = None):
        """Persist a verbose error report to a text file."""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"sanity_test_report_{timestamp}.txt"

        with open(output_path, 'w') as fh:
            fh.write(f"Sanity Test Report for Collection: {self.collection_name}\n")
            fh.write(f"Generated: {datetime.now()}\n")
            fh.write("=" * 80 + "\n\n")

            for error_type, error_list in self.errors.items():
                if error_list:
                    fh.write(f"{error_type.upper().replace('_', ' ')}:\n")
                    fh.write("-" * 40 + "\n")
                    for err in error_list:
                        fh.write(f"{err}\n")
                    fh.write("\n")

        print(f"📁 Detailed report saved to: {output_path}")


@hydra.main(version_base=None, config_path="./conf", config_name="config")
def run_sanity_test(cfg: DictConfig):
    """
    Hydra entry point for the sanity test.

    Args:
        cfg: Hydra configuration object containing database settings.
    """
    cfg.memory.memory_store = "external_baseline-4.1-subset1-cueindex"
    client = PersistentClient(path=cfg.memory.persist_path)

    tester = MemoryLinkSanityTester(client, cfg.memory.collection_name)

    outcome = tester.run_comprehensive_test(user_id="Melanie_0")

    total_errors = (
        outcome['stats']['broken_linked_memories']
        + outcome['stats']['broken_cue_indices']
        + outcome['stats']['orphaned_cues']
        + outcome['stats']['orphaned_primaries']
    )

    if total_errors > 0:
        tester.save_detailed_report()

    return outcome


if __name__ == "__main__":
    run_sanity_test()
