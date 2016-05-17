# Running this on csc

login: taito-gpu.csc.fi

# Basic setup only needs to be done once

module load python-env/3.4.0
pip3 install --user virtualenv

# create the virtual env
python3 -m virtualenv $USERAPPL/venv_keras_py3
# and enter it
source $USERAPPL/venv_keras_py3/bin/activate
pip3 install keras pycuda sympy  #Maybe something else?

# Make a script in $USERAPPL/activate_gpu.sh like so
module purge
module load StdEnv
module load gcc/4.9.3
module load git
module load cuda
#module load boost #not sure why this doesn't work, but isn't needed I guess
module load openblas
module load hdf5-serial
module load python-env/3.4.1
source $USERAPPL/venv_keras_py3/bin/activate

# now "sbatch train_bsub.sh" should work

