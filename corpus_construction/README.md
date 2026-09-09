# Processed-Corpus Construction

These staged utilities were used to construct the fixed textual
folder/document registry for the RAGBench HotpotQA experiments. The input is
`ragbench_test.jsonl`; PDF parsing is not part of this workflow.

Corpus construction is completed before evaluation. MiloNet uses the same
registry for every query and does not change document placement at query time.

## Construction stages

1. `RAGBench_test.py` extracts the documents in `ragbench_test.jsonl` to
   `txt_docs/`. It embeds the documents with `all-MiniLM-L6-v2`, forms five
   KMeans groups with up to three initial subgroups per group, and writes the
   initial hierarchy to `Ori-docs/`.
2. Place the initial `Ori-docs/` hierarchy at `Original_documents/`, then
   initialize MiloNet. Initialization creates the corresponding document
   stores, indexes, summaries, and `processed_files_map.json`.
3. Export the document-summary records from Chroma as
   `embedding_metadata.json`. `convertdb2Json.py` converts this file to
   `doc_summary_dict.json`.
4. `organize_document_hierarchy.py` organizes the document summaries into a
   semantic hierarchy. The folder-merging code limits each folder to at most
   three file nodes.
   After the candidate hierarchy is generated, save it as
   `organized_structure_duplicates.json`. Duplicate removal then writes
   `organized_structure_processed.json`, which is subsequently checked against
   `doc_summary_dict.json`.
5. Save the hierarchy to be checked as `organized_structure_no_extras.json`,
   then run `validateStructure.py`. It reports document IDs that are missing
   from the hierarchy or absent from `doc_summary_dict.json`.
6. `folder_structure_constructor.py` resolves the document IDs through
   `processed_files_map.json` and materializes the hierarchy in
   `New_Folder_Structure/`.
   The map stores original relative paths as keys and processed Markdown paths
   as values. The constructor matches a compact document ID to the processed
   path basename and then recovers the original relative path.
7. After placing the finalized tree at `Original_documents/`, run
   `shorter_name.py` to assign compact numeric folder and file names. The
   original-to-compact mapping is written to `name_mapping.json`.
8. Initialize MiloNet over the finalized `Original_documents/` tree to create
   the processed registry used for inference.

The utilities are run stage by stage and exchange intermediate files through
the current working directory. Run them in a separate working directory
containing the required inputs and intermediate files.

## Reproducibility notes

The KMeans calls in `RAGBench_test.py` do not set `random_state`, and the
organization stage makes model-service calls. A new construction can therefore
produce different folder assignments or names. The reported experiments use a
single registry built before evaluation and held fixed for all queries.
