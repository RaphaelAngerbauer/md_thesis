rule gfp_deg_fluorometry:
    input: spark = "src/data/gfp_deg/raw/fluorometry/1/Spark.xlsx",
        group = "src/data/gfp_deg/raw/fluorometry/1/Group.xlsx",
        bradford = "src/data/gfp_deg/raw/fluorometry/1/Bradford.xlsx",
        bradlab = "src/data/gfp_deg/raw/fluorometry/1/BradLab.xlsx",
    output: "src/data/gfp_deg/processed/fluorometry/1/results.csv"
    shell: """
    python src/processing/fluorometry_1d.py -s {input.spark} -b {input.bradford} -sl {input.group} -bl {input.bradlab} -o {output}
    """
