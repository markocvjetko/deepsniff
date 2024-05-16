
/home/michielc-haeslerlab/scripts/flirsniff/singularity/nosetrack_deeplabcut_analysis_marko.py

To be run in this container: /home/michielc-haeslerlab/scripts/flirsniff/singularity/nose_track.sif

Check the python script for the required arguments.

```bash
module load singularity/3.5

singularity exec --nv --bind /mnt:/mnt /home/michielc-haeslerlab/scripts/flirsniff/singularity/nose_track.sif python3 /home/michielc-haeslerlab/scripts/flirsniff/singularity/nosetrack_deeplabcut_analysis.py
```

