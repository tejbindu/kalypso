"""Typed errors raised by block execution components."""


class BlockFilterError(Exception):
    """Base class for block-filter failures."""


class BlockConfigurationError(BlockFilterError, ValueError):
    """Invalid block settings or an incompatible block plan."""

    def __init__(self, message: str, operator_position: int | None = None):
        self.operator_position = operator_position
        self.reason = message
        location = (
            f" at operator position {operator_position}"
            if operator_position is not None
            else ""
        )
        super().__init__(f"Invalid block configuration{location}: {message}")


class BlockSizingError(BlockFilterError, ValueError):
    """A row or prepared block cannot fit configured limits."""

    def __init__(self, message: str, row_id: str | None = None):
        self.row_id = row_id
        self.reason = message
        location = f" for row {row_id}" if row_id is not None else ""
        super().__init__(f"Block sizing failed{location}: {message}")


class BlockOutputError(BlockFilterError):
    """A model response failed block protocol validation."""

    def __init__(self, message: str, block_id: str | None = None):
        self.block_id = block_id
        self.reason = message
        location = f" for block {block_id}" if block_id is not None else ""
        super().__init__(f"Invalid block output{location}: {message}")


class BlockExecutionError(BlockFilterError):
    """An engine or block-pipeline operation failed."""

    def __init__(self, message: str, block_id: str | None = None):
        self.block_id = block_id
        self.reason = message
        location = f" for block {block_id}" if block_id is not None else ""
        super().__init__(f"Block execution failed{location}: {message}")
