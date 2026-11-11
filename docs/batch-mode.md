# Batch Mode (quantify without the GUI)

`SECQUOIA-batch` quantifies tracked data from the command line, without opening any window and without curation. It does exactly what the [loading window](loading-window.md) does when you choose the `All` loading format and click `Run`: it loads the tracking data, then loads, background-corrects, masks and quantifies every selected position.

Use it to:

- quantify all positions of an experiment while you do something else,
- process several experiments one after another with a script,
- run on a server or cluster that has no screen.

The results are the same files the GUI writes, in the same project folder. You can open the project in SECQUOIA afterwards (`File → Open a project`) and curate it as usual.

## Running it

Activate the environment SECQUOIA is installed in, then run `SECQUOIA-batch` with the experiment folder and the tracking folder:

```bash
conda activate SECQUOIA
SECQUOIA-batch /path/to/250615MA40 --tracking-folder Analysis/Tracking_tTt/250615MA40
```

All other settings have the same defaults as the loading window. Before a long run, add `--dry-run` to check what was detected. It prints the same summary as the GUI's Loading tab and stops without writing anything.

> **Note:** The `SECQUOIA-batch` command is created when SECQUOIA is installed. If you updated with `git pull` only, run `pip install .` (or `pip install -e ".[testing]"` for a developer install) once to get it. Until then, `python -m SECQUOIA.cli.batch` works the same way.

`SECQUOIA-batch --help` lists every option.

### Try it on the demo data

After unzipping the [demo dataset](../examples/README.md), from the repository root:

```bash
SECQUOIA-batch examples/data/250615MA40 \
    --tracking-folder Analysis/Tracking_tTt/250615MA40 \
    --user MA --real-time
```

This quantifies both positions with both segmentations and all five channels, and writes the results to `examples/data/250615MA40/Analysis/SECQUOIA_files_tTt/Project_1/`.

## From the loading window to arguments

Each field of the loading window has a matching argument. A field you leave out keeps the GUI's default.

### Tracking tab

| Loading window | Argument | Default |
| --- | --- | --- |
| 1. Project name | `--project-name NAME` | `Project_1` |
| 2. User initials | `--user INITIALS` | empty |
| 3. Experiment folder | first argument (required) | |
| 4. Image format | `--image-format {png,tif,jpg}` | `png` |
| 5. Tracking input format | `--tracking-format {btrack,CTC,tTt,Ultrack}` | `tTt` |
| 6. Tracking folder | `--tracking-folder PATH` (required) | |
| 7. `Load experiment` | runs automatically | |
| 8. Positions in the selection tree | `--positions 1-3,5` | every position with tracking data |
| 8. Identifications (Tree-IDs) in the selection tree | `--tree-ids ID [ID ...]` or `--tree-ids-file FILE` | all |

- A relative `--tracking-folder` is looked up from the current folder first, then inside the experiment folder. So `Analysis/Tracking_tTt/<experiment>` works from anywhere.
- tTt tracking also needs the `*_TATexp.xml` file in the experiment folder, as in the GUI.
- `--positions` takes numbers and ranges: `2`, `1-4`, `1-3,7,9-12`.
- Tree-IDs are the `Identification` values, for example `250615MA40-p0001-003`. Use `--list-tree-ids` to print them all (one `Position<TAB>Tree-ID` per line), then pick the ones you want. `--tree-ids-file` reads one Tree-ID per line.
- When both `--positions` and Tree-IDs are given, only Tree-IDs inside those positions are kept.

### Segmentation tab

| Loading window | Argument | Default |
| --- | --- | --- |
| 1. Segmentation folders | `--segmentation NAME [NAME ...]` | all `Segmentation*` folders |
| 2. BaSiC background (checkbox and folder) | `--basic` or `--basic BaSiC_folder_name` | off |
| 3. Real time (checkbox and `images*.csv`) | `--real-time` or `--real-time images.csv` | off |
| 4. Minimum cell size | `--min-cell-size PX` | `5` |
| 5. Maximum distance from tracking point to mask | `--max-distance PX` | `20` |

- Segmentation folders are given by name, as they appear in the `Analysis` folder. Masks are numbered `M1`, `M2`, … in alphabetical order of the folder names, exactly as in the GUI, whatever order you type them in. The summary shows which folder became which mask.
- `--basic` and `--real-time` without a name use the first folder or file found, as the GUI does.

### Channel Selection tab

| Loading window | Argument | Default |
| --- | --- | --- |
| 1. Channels | `--channels w00 w02 ...` | all detected channels |
| 2. Minimum time point | `--t-min T` | `1` |
| 3. Maximum time point | `--t-max T` | last time point found |
| 4. Time interval in seconds | `--interval SECONDS` | `300` |
| 5. Start position for curation | `--start-position N` | first quantified position |

`--start-position` only decides which position SECQUOIA shows first when you open the project in the GUI.

### Loading tab

| Loading window | Argument |
| --- | --- |
| Loading format | always `All` (every selected position is quantified) |
| `Run` | running the command |
| Summary | printed before the run starts; `--dry-run` prints it and stops |
| "Project already exists. Overwrite?" | `--overwrite` |

If the project folder already exists, the run stops before doing anything. Choose another `--project-name`, or add `--overwrite`. Like the GUI's `Overwrite` button, `--overwrite` **permanently deletes** the old project folder, including any curation saved in it.

### Other options

| Argument | What it does |
| --- | --- |
| `--dry-run` | Load the tracking data, print the summary and exit. Nothing is written. |
| `--list-tree-ids` | Print every selected Position and Tree-ID and exit. |
| `-v`, `--verbose` | Show every processing step in the terminal. |
| `-q`, `--quiet` | Show only warnings and errors. |

## What gets written

Everything goes to `<experiment>/Analysis/SECQUOIA_files_<tracking format>/<project name>/`. See [output folders](data-formats.md#output-folders-created-by-secquoia) and the [CSV columns](data-formats.md#exported-csv-files-by-secquoia) for the file contents.

- `Updated_CSV_File.csv`: all positions in one table. This is the file to analyse.
- `SECQUOIA_p0001_t00001-00026_m2_ch5.csv`, …: one CSV per quantified position.
- `project_metadata.json`: all settings, so the project can be reopened in SECQUOIA with `File → Open a project`.

The full log is appended to `<experiment>/Analysis/SECQUOIA.log`, the same log the GUI writes.

The quantification is uncurated. Each tracking point is matched to the nearest mask within `--max-distance`. Points without a mask in range keep empty measurement values. Set the minimum cell size and maximum distance with care, and check a few Tree-IDs in the GUI before relying on the numbers.

## Differences from the GUI

- Only the `All` loading format is available. Curation always happens in the GUI.
- If no segmentation folders or channels are given, all of them are used. The GUI makes you tick them.
- At the end of a GUI run, only the per position CSV of the position it shows is rewritten with the final table. `SECQUOIA-batch` rewrites the CSV of every quantified position, so each one matches its rows in `Updated_CSV_File.csv`. `Updated_CSV_File.csv` is the same as the GUI's.
- If a position fails (for example, unreadable images), the run continues with the next one. The failed position gets no per position CSV, and the run ends with exit code `1`.

## Exit codes

| Code | Meaning |
| --- | --- |
| `0` | Every selected position was quantified. |
| `1` | The run finished, but at least one position failed. The log names them. |
| `2` | Nothing was run: a wrong argument, a missing folder, or an existing project without `--overwrite`. The message says what to fix. |

## Processing several experiments

The exit codes make it easy to script. For example, in a terminal on macOS or Linux:

```bash
for exp in /data/2506*MA*; do
    name=$(basename "$exp")
    SECQUOIA-batch "$exp" \
        --tracking-folder "Analysis/Tracking_tTt/$name" \
        --project-name Batch_1 --max-distance 20 --min-cell-size 5 \
        || echo "Problem with $name"
done
```

`SECQUOIA-batch` needs no screen, so it also runs over SSH or as a cluster job. It needs the same memory and temporary disk space per position as the GUI (see [system requirements](../README.md#system-requirements)).
