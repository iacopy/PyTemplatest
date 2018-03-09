#!/usr/bin/env python
"""
With arguments ask if git status is exactly mathes the query,
else prints status and returns 0 exit code only in case of
no files in index nor untracked.
"""
# Standard Library
import subprocess
import sys
from collections import defaultdict
from typing import Dict
from typing import List

ST_CLEAN = 'clean'
ST_INDEX = 'index'
ST_DIRTY = 'dirty'
EXIT_DICT = {ST_CLEAN: 0, ST_INDEX: 1, ST_DIRTY: 2}


def get_git_status() -> Dict[str, List[str]]:
    """
    Call git status and parse results in a dictionary.
    """
    result = defaultdict(list)  # type: dict
    output_lines = subprocess.check_output(
        ['git', 'status', '--porcelain']).decode('utf-8').splitlines()
    for line in output_lines:
        index, worktree, filename = line[0].strip(), line[1].strip(), line[3:]
        if index:
            result['index'].append(filename)
        if worktree:
            result['worktree'].append(filename)
    return dict(result)


def str_status(git_status: Dict[str, List[str]]) -> str:
    """
    Return a string ('clean', 'index' or 'dirty') as a
    syntesis of git status.

    >>> str_status({})
    'clean'
    >>> str_status({'worktree': ['foo']})
    'dirty'
    >>> str_status({'index': ['a', 'b', 'c']})
    'index'

    """
    if not git_status:
        return ST_CLEAN

    worktree = git_status.get('worktree')
    index = git_status.get('index')
    if worktree:
        return ST_DIRTY
    assert index, 'Programming error: index should be not empty but it is'
    return ST_INDEX


def explain_errors(query, git_status):
    """
    Explicitly describes the git status error, given the `query`.
    """
    error_lines = []
    if query == 'clean':
        for place in ('worktree', 'index'):
            files = git_status.get(place)
            if files:
                msg = ('ERROR: {} should be clean, '
                       'but there are files in:'.format(place))
                error_lines.append(msg)
                for file_path in files:
                    error_lines.append('\t' + file_path)
    elif query == 'index':
        worktree_files = git_status.get('worktree')
        if worktree_files:
            error_lines.append(
                'ERROR: worktree should be clean, but there are files in it:')
            for file_path in worktree_files:
                error_lines.append('\t' + file_path)
            error_lines.append('HINT: git add -u  # stage modified files')
            error_lines.append('HINT: git add --all  # stage everything')
            error_lines.append(
                'HINT: just clean #  remove (!) nontracked files')
        if not git_status.get('index'):
            error_lines.append('ERROR: there are no files in the index.')
            if worktree_files:
                error_lines.append('HINT: git add -u  # stage modified files')
            else:
                error_lines.append('HINT: Make some changes to your codebase')

    return error_lines


def main(args: List[str]) -> int:
    """
    Call git status and convert it as an exit code.

    0 = clean: worktree and index both empty
    1 = index: index not empty and worktree empty
    2 = dirty: worktree not empty

    With arguments, return 0 only in case of status matches the query.
    Allowed queries: 'clean', 'index' or 'dirty'.
    """
    git_status = get_git_status()
    status = str_status(git_status)
    query = args[0] if args else ''
    if query:
        # Return 0 if status matches the query, 1 otherwise
        if query != status:
            print('\n'.join(explain_errors(query, git_status)))
        return int(query != status)

    # print status and return the exit code related to the status
    # Handle verbosity?
    print(status)
    return EXIT_DICT[status]


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
