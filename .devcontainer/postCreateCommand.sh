git config --global safe.directory '*'
git config --global core.editor "code --wait"
git config --global pager.branch false

# install precommit hooks
pre-commit install

# Install metaseq and dependencies
conda run -n ptca pip install --user -e ".[dev,docs]"
conda run -n ptca python setup.py develop --user

# Install docs dependencies and generate docs
cd docs
make html
cd ..
