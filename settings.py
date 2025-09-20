from pydantic.dataclasses import dataclass


@dataclass
class Setting:
    DEBUG_SHOW: int = 0
    DEBUG_SAVE_IMG: int = 1
    DEBUG_COUNT_OBJ: int = 1
    LOG_DIRS: str = "logs"

    MIN_AREA: int = 50
    step: int = 0


settings = Setting()