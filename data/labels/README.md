# Reviewed labels

Every CSV file in this folder is used for training, together with
`data/seed_sample.csv`, both by `python -m ml.prepare_data` and when the Docker
image is built on the server.

Put only **reviewed, anonymised** files here, usually the user feedback exported
with `bash deploy/export-feedback.sh` on the server (or `python -m ml.export_feedback`).

Rules:

- Use the same columns as the seed sample (see `data/LABELING_GUIDE.md`), plus an
  optional `keep` column.
- To leave a row out, set `keep` to `no` instead of deleting it. Rows stay in the file,
  so the next export knows they were already reviewed and does not offer them again.
- Before committing a file, read through the `text` column for any names or numbers
  the automatic anonymisation missed, and remove them.
- Never change or reuse an `id`.
