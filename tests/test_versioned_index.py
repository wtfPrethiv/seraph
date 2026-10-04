from __future__ import annotations

from pathlib import Path
import subprocess
import tempfile
import unittest

from seraph.index import VersionedIndex


def git(repo: Path, *args: str) -> str:
    result = subprocess.run(["git", "-C", str(repo), *args], check=True,
                            capture_output=True, text=True)
    return result.stdout.strip()


class VersionedIndexTests(unittest.TestCase):
    def test_changed_file_is_reparsed_and_unchanged_content_is_reused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            git(repo, "init", "-q")
            git(repo, "config", "user.email", "test@example.com")
            git(repo, "config", "user.name", "Seraph Test")
            (repo / "route.py").write_text('def route():\n    return "rocket"\n')
            (repo / "stable.py").write_text('def stable():\n    return "anchor"\n')
            git(repo, "add", ".")
            git(repo, "commit", "-qm", "initial")
            old_commit = git(repo, "rev-parse", "HEAD")

            with VersionedIndex(repo) as index:
                first = index.index_commit(old_commit)
                self.assertEqual(first.parsed_files, 2)
                self.assertEqual(first.total_chunks, 2)
                self.assertEqual(index.search("rocket", old_commit)[0].chunk.path, "route.py")

                (repo / "route.py").write_text('def route():\n    return "planet"\n')
                git(repo, "add", "route.py")
                git(repo, "commit", "-qm", "change route")
                new_commit = git(repo, "rev-parse", "HEAD")
                second = index.index_commit(new_commit)

                self.assertEqual(second.parsed_files, 1)
                self.assertEqual(second.reused_chunks, 1)
                self.assertEqual(second.total_chunks, 2)
                self.assertEqual(index.search("rocket", new_commit), [])
                self.assertEqual(index.search("planet", new_commit)[0].chunk.path, "route.py")
                self.assertEqual(index.search("rocket", new_commit, include_history=True)[0].chunk.commit,
                                 old_commit)

                old_stable = next(chunk for chunk in index.iter_chunks(old_commit)
                                  if chunk.path == "stable.py")
                new_stable = next(chunk for chunk in index.iter_chunks(new_commit)
                                  if chunk.path == "stable.py")
                self.assertEqual(old_stable.content_hash, new_stable.content_hash)
                self.assertNotEqual(old_stable.occurrence_id, new_stable.occurrence_id)

    def test_invalid_python_falls_back_to_file_chunk(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            git(repo, "init", "-q")
            git(repo, "config", "user.email", "test@example.com")
            git(repo, "config", "user.name", "Seraph Test")
            (repo / "draft.py").write_text("def unfinished(\n")
            git(repo, "add", ".")
            git(repo, "commit", "-qm", "draft")

            with VersionedIndex(repo) as index:
                index.index_commit()
                chunk = next(index.iter_chunks())
                self.assertEqual(chunk.kind, "file")
                self.assertEqual(chunk.path, "draft.py")


if __name__ == "__main__":
    unittest.main()
