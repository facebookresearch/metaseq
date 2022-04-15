# Contributing to metaseq
We want to make contributing to this project as easy and transparent as
possible.

## Pull Requests
We actively welcome your pull requests.

1. Fork the repo and then clone the forked repository.
   * See this [github guide](https://guides.github.com/activities/forking/) on forking for more info.
   * **If you have already cloned the repo directly and committed changes, follow the steps in the [section below](#moving-changes-youve-committed-to-a-fork)**
2. Create your branch from `main`.
3. Set up your environment and run `pre-commit install` once.
4. Make your changes
5. If you've added code that should be tested, add tests.
6. If you've changed APIs, update the documentation.
7. (Optional) Ensure the test suite passes. Run `python -m pytest -m unit`.
8. If you've added a new dataset, you should also run
   `python -m pytest -m data`. Copy-paste the output into a comment in your PR.
9. If you haven't already, complete the Contributor License Agreement ("CLA").
10. Link [CircleCI](https://circleci.com/vcs-authorize/) to your github account
    if you haven't done so previously (and make sure the CircleCI tests run
    successfully on the PR after you push your changes).
11. Push your changes!
12. Once the PR is accepted and CI is passing, we will merge the PR for you.

### Moving changes you've committed to a fork
1. Fork the repo
2. In your local repo, rename your origin remote to upstream
   ```
   git remote rename origin upstream
   ```
3. Point origin to the forked repo (instead of to the original repo)
   ```
   git remote add origin git@github...<FORK>
   ```
4. Fetch from the new origin
   ```
   git fetch origin
   ```
5. Make your local branch track the remote branch (of the forked repo)
   ```
   git branch --set-upstream-to origin/main main

### Pre-commit hooks
In order to ensure your code lints, there are pre-commit hooks configured in the repository which you can install.
After installation, they will automatically run each time you commit.
An abbreviated guide is given below; for more information, refer to [the offical pre-commit documentation](https://pre-commit.com/).

```
pip install pre-commit
pre-commit install
```

## Contributor License Agreement ("CLA")
In order to accept your pull request, we need you to submit a CLA. You only need
to do this once to work on any of Meta's open source projects.

Complete your CLA here: <https://code.facebook.com/cla>

## Issues
We use GitHub issues for general feature discussion, Q&A and public bugs tracking.
Please ensure your description is clear and has sufficient instructions to be able to
reproduce the issue or understand the problem.

## License
By contributing to metaseq, you agree that your contributions will be licensed
under the LICENSE file in the root directory of this source tree.