import asyncio
import os
from asyncio import Semaphore
from typing import Optional

from rich.progress import Progress, TaskID

from .rich_console import console, create_progress


class Parallel:
    """Wrapper around semaphore to limit concurrency"""

    __jobs: list
    __semaphore: Semaphore
    __progress: Progress
    __task_id: TaskID
    __immediate: bool
    __owns_progress: bool
    __owns_task: bool

    def __init__(
        self,
        title: str = None,
        unit: str = "file",
        concurrency: int = os.cpu_count(),
        immediate: bool = False,
        progress: Optional[Progress] = None,
        task_id: Optional[TaskID] = None,
        **_kwargs,
    ):
        self.__jobs = []
        self.__semaphore = Semaphore(concurrency)
        self.__progress = progress or create_progress()
        self.__owns_progress = progress is None

        if self.__owns_progress:
            self.__progress.start()

        description = title or "Processing"
        if task_id is not None:
            self.__task_id = task_id
            self.__owns_task = False
            self.__progress.update(task_id, description=description, unit=unit)
        else:
            self.__task_id = self.__progress.add_task(
                description,
                total=0,
                unit=unit,
            )
            self.__owns_task = True

        self.__immediate = immediate

    async def __run(self, func: callable, *args, **kwargs):
        """Runs the given function with limited concurrency."""
        async with self.__semaphore:
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                self.__progress.advance(self.__task_id, 1)

    def run(self, func: callable, *args, **kwargs):
        """Runs the given function with limited concurrency."""
        job = self.__run(func, *args, **kwargs)
        if self.__immediate:
            job = asyncio.create_task(job)
        self.__jobs.append(job)

    def log(self, message: str):
        """Log a message using the shared console."""
        console.log(message)

    def count_jobs(self):
        """Returns the number of jobs in the queue"""
        return len(self.__jobs)

    async def wait(self):
        """Run the collected tasks."""
        total = self.count_jobs()
        self.__progress.update(
            self.__task_id,
            total=total,
            completed=0,
            visible=total > 0,
        )

        if total == 0:
            if self.__owns_task:
                self.__progress.update(self.__task_id, visible=False)
            if self.__owns_progress:
                self.__progress.stop()
            return

        await asyncio.gather(*self.__jobs)

        if self.__owns_task:
            self.__progress.update(self.__task_id, visible=False)

        if self.__owns_progress:
            self.__progress.stop()

    @property
    def task_id(self) -> TaskID:
        """Expose the underlying progress task identifier."""
        return self.__task_id

    @property
    def progress(self) -> Progress:
        """Expose the underlying progress instance."""
        return self.__progress


