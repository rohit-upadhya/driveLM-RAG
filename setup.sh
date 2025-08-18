unset PYTHONPATH
poetry shell
poetry install
export PYTHONPATH="$PYTHONPATH:$(pwd)"
echo "Python path set to : $PYTHONPATH"