"""Minimal settings and observer pattern, vendored from sdeconv.core."""

import torch
from abc import ABC, abstractmethod


class SSettingsContainer:
    """Container for the SDeconv library settings."""
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def get_device(self) -> str:
        return self.device


class SSettings:
    """Singleton to access the Settings container."""
    __instance = None

    @staticmethod
    def instance():
        if SSettings.__instance is None:
            SSettings.__instance = SSettingsContainer()
        return SSettings.__instance


class SObserver(ABC):
    @abstractmethod
    def notify(self, message: str):
        raise NotImplementedError

    @abstractmethod
    def progress(self, value: int):
        raise NotImplementedError


class SObservable:
    def __init__(self):
        self._observers = []

    def add_observer(self, observer: SObserver):
        self._observers.append(observer)

    def notify(self, message: str):
        for obs in self._observers:
            obs.notify(message)

    def progress(self, value):
        for obs in self._observers:
            obs.progress(value)
