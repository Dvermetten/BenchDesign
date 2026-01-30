This repository contains the getting started documents for the BenchDesign competition, hosted at GECCO 2026. 
For general information regarding the competition, please consult [this page](https://iohprofiler.github.io/competitions/benchdesign)

To create your submission, you should fill the problem_suite file with 2 dictionaries, containing your problems and their problem dimensionalities respectively. 
Then, you can call the evaluation file to calculate the performance of this created suite. This evaluation procedure runs 5 selected algorithms on your suite, and then makes use of the metric described in \[1\] to determine the ranking differences between their fixed-budget performances.


### Quickstart

1. First clone this repository.

2. Setup your environment for example using Python 3.10.


    ```bash
    python3.10 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

3. Design your functions in problem_suite.py.

4. Run `python evaluation.py`to validate your problem suite.


\[1\] L. Stripinis, J. Kůdela and R. Paulavičius, "Two Novel Instance Selection Methods Combining Algorithm Performance and Landscape Analysis: A Comparative Study in Continuous Optimization," in IEEE Transactions on Cybernetics, doi: 10.1109/TCYB.2025.3625095
