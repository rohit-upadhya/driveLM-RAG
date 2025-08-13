unset PYTHONPATH
poetry shell

export PYTHONPATH="$PYTHONPATH:$(pwd)"
echo "Python path set to : $PYTHONPATH"