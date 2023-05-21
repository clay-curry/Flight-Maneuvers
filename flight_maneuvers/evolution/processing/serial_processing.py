
""" Implements a simple wrapper for the serial processing of items.
"""

from typing import Callable, List, Sequence

from flight_maneuvers.evolution.processing.base_scheduler import ProcessingScheduler
from flight_maneuvers.evolution.processing.base_scheduler import TProcItem, TProcResult


class SerialProcessingScheduler(ProcessingScheduler):
    """ Simple wrapper for the serial processing of items.

    This is scheduler is just a wrapper for the serial processing of items,
    i.e., the processing of one item at a time. It doesn't involve any explicit
    parallel processing.
    """

    def run(self,
            items: Sequence[TProcItem],
            func: Callable[[TProcItem], TProcResult]) -> List[TProcResult]:
        """ Sequentially processes the input items.

        Args:
            items (Sequence[TProcItem]): Iterable containing the items to be
                processed.
            func (Callable[[TProcItem], TProcResult]): Callable (usually a
                function) that takes one item :attr:`.TProcItem` as input and
                returns a result :attr:`.TProcResult` as output. Generally,
                :attr:`.TProcItem` is an individual in the population and
                :attr:`.TProcResult` is the individual's fitness. If additional
                arguments must be passed to the callable you want to use, it's
                possible to use Python's :mod:`functools.partial` or to just
                wrap it with a simple function.

        Returns:
            A list containing the results of the processing of each item. It is
            guaranteed that the ordering of the items in the returned list
            follows the order in which the items are yielded by the iterable
            passed as argument.
        """
        return [func(item) for item in items]
