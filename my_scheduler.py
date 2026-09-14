import logging
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.sched.output import SchedulerOutput

logger = logging.getLogger(__name__)


class StepExclusivePrefillFirstScheduler(Scheduler):
    """Strategy A: prefill-first.

    Mirror of Strategy B: save -> clear -> super() -> restore.
    Temporarily clear self.running so only prefill requests are scheduled.
    """
    def schedule(self) -> SchedulerOutput:
        if len(self.waiting) > 0:
            saved_running = list(self.running)
            self.running.clear()
            try:
                output = super().schedule()
            finally:
                self.running.extend(saved_running)
            logger.info(f"[Strategy A] PREFILL step, scheduled={output.num_scheduled_tokens}")
            return output
        else:
            output = super().schedule()
            logger.info(f"[Strategy A] DECODE step, scheduled={output.num_scheduled_tokens}")
            return output


class PureDecodeFirstScheduler(Scheduler):
    """Strategy B: Pure Decode-First.

    Mirror of Strategy A: when running non-empty, only schedule decode.
    Temporarily clear self.waiting to prevent prefill from mixing in.
    """

    def schedule(self) -> SchedulerOutput:
        if len(self.running) > 0:
            saved_waiting = list(self.waiting)
            self.waiting.clear()
            try:
                output = super().schedule()
            finally:
                self.waiting.extend(saved_waiting)
            logger.info(f"[Strategy B] DECODE step, scheduled={output.num_scheduled_tokens}")
            return output
        else:
            output = super().schedule()
            logger.info(f"[Strategy B] PREFILL step, scheduled={output.num_scheduled_tokens}")
            return output
