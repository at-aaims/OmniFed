# Copyright (c) 2025, Oak Ridge National Laboratory.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
General-purpose string matching utilities.

These functions return callable matchers that can be used for pattern matching
in various contexts throughout FLORA (metric formatting, configuration filtering, etc.).
"""

import re
from typing import Callable

# ======================================================================================
# STRING MATCHER FUNCTIONS
# ======================================================================================

def contains(keyword: str) -> Callable[[str], bool]:
    """
    Create a matcher that checks if a string contains the given keyword.
    
    Args:
        keyword: The substring to search for (case-insensitive)
        
    Returns:
        A function that returns True if the input string contains the keyword
        
    Example:
        matcher = contains('loss')
        matcher('train/loss')  # True
        matcher('accuracy')   # False
    """
    return lambda name: keyword in name.lower()


def startswith(prefix: str) -> Callable[[str], bool]:
    """
    Create a matcher that checks if a string starts with the given prefix.
    
    Args:
        prefix: The prefix to match (case-insensitive)
        
    Returns:
        A function that returns True if the input string starts with the prefix
        
    Example:
        matcher = startswith('train/')
        matcher('train/loss')    # True
        matcher('eval/loss')     # False
    """
    return lambda name: name.lower().startswith(prefix.lower())


def endswith(suffix: str) -> Callable[[str], bool]:
    """
    Create a matcher that checks if a string ends with the given suffix.
    
    Args:
        suffix: The suffix to match (case-insensitive)
        
    Returns:
        A function that returns True if the input string ends with the suffix
        
    Example:
        matcher = endswith('/loss')
        matcher('train/loss')  # True
        matcher('train/acc')   # False
    """
    return lambda name: name.lower().endswith(suffix.lower())


def exact(pattern: str) -> Callable[[str], bool]:
    """
    Create a matcher that checks for exact string equality.
    
    Args:
        pattern: The exact string to match (case-insensitive)
        
    Returns:
        A function that returns True if the input string exactly equals the pattern
        
    Example:
        matcher = exact('accuracy')
        matcher('accuracy')      # True
        matcher('train/accuracy') # False
    """
    return lambda name: name.lower() == pattern.lower()


def regex(pattern: str) -> Callable[[str], bool]:
    """
    Create a matcher using regular expressions.
    
    Args:
        pattern: Regular expression pattern (case-insensitive by default)
        
    Returns:
        A function that returns True if the pattern matches the input string
        
    Example:
        matcher = regex(r'grad(_norm|ient)?')
        matcher('grad_norm')   # True
        matcher('gradient')    # True
        matcher('accuracy')    # False
    """
    compiled = re.compile(pattern, re.IGNORECASE)
    return lambda name: bool(compiled.search(name))


def any_of(*patterns: str) -> Callable[[str], bool]:
    """
    Create a matcher that checks if a string contains any of the given patterns.
    
    Args:
        *patterns: Variable number of substrings to search for (case-insensitive)
        
    Returns:
        A function that returns True if the input string contains any of the patterns
        
    Example:
        matcher = any_of('accuracy', 'precision', 'recall')
        matcher('train/accuracy')  # True
        matcher('eval/precision')  # True  
        matcher('loss')           # False
    """
    return lambda name: any(pattern in name.lower() for pattern in patterns)


def all_of(*patterns: str) -> Callable[[str], bool]:
    """
    Create a matcher that checks if a string contains all of the given patterns.
    
    Args:
        *patterns: Variable number of substrings that must all be present
        
    Returns:
        A function that returns True if the input string contains all patterns
        
    Example:
        matcher = all_of('train', 'loss')
        matcher('train/loss')     # True
        matcher('train/accuracy') # False
        matcher('eval/loss')      # False
    """
    return lambda name: all(pattern in name.lower() for pattern in patterns)


def none_of(*patterns: str) -> Callable[[str], bool]:
    """
    Create a matcher that checks if a string contains none of the given patterns.
    
    Args:
        *patterns: Variable number of substrings that must not be present
        
    Returns:
        A function that returns True if the input string contains none of the patterns
        
    Example:
        matcher = none_of('test', 'debug')  
        matcher('train/loss')    # True
        matcher('test/accuracy') # False
    """
    return lambda name: not any(pattern in name.lower() for pattern in patterns)