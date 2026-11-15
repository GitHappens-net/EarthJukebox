"""Beat-synchronized 75 BPM C-Minor sequencer engine inspired by Veridis Quo."""

from __future__ import annotations

import threading
import time
from typing import Optional

from backend.earth_jukebox.audio.backends import AudioBackend, create_best_audio_backend
from backend.earth_jukebox.audio.theory import (
    ARP_CHANNEL,
    BASS_CHANNEL,
    BEAT_DURATION_SEC,
    BPM,
    C_MINOR_PITCH_CLASSES,
    CHURCH_ORGAN_CHANNEL,
    COLOR_CHANNEL_PROGRAM,
    COLOR_PITCH_MAP,
    DRUM_CHANNEL,
    DRUM_CLAP,
    DRUM_CONGA_HI,
    DRUM_HAT_CLOSED,
    DRUM_HAT_OPEN,
    DRUM_KICK,
    DRUM_LOW_TOM,
    DRUM_RIM,
    DRUM_SHAKER,
    DRUM_TAMBOURINE,
    CHANNEL_PAN,
    CHANNEL_VOLUME,
    COLOR_INSTRUMENT_NAME,
    HARMONIC_PROGRESSION,
    NOTE_NAMES,
    PAD_CHANNEL,
    PLUCK_CHANNEL,
    SIXTEENTH_DURATION_SEC,
    VERIDIS_QUO_PROGRESSION,
    WARM_PAD_CHANNEL,
    AnalysisResult,
    AnalysisSmoother,
    NoteEvent,
    PAD_ROLES,
    StepPlan,
    StepSequencer,
    active_layers,
    compute_expression_ccs,
    intensity_to_velocity,
    intensity_to_volume,
    midi_to_freq,
    midi_to_note_name,
    plan_ambient_drums,
    plan_baroque_arpeggio,
    plan_bass_step,
    plan_chord_voicing,
    plan_musical_step,
    plan_spottiness_accents,
    select_rhythm,
    should_trigger_data_step,
    snap_to_c_minor,
)


class CMinorAudioEngine:
    """Locked 75 BPM 16th-note Veridis Quo-inspired C-Minor polyphonic sequencer."""

    BPM: float = BPM
    BEAT_DURATION_SEC: float = BEAT_DURATION_SEC
    SIXTEENTH_DURATION_SEC: float = SIXTEENTH_DURATION_SEC

    C_MINOR_PITCH_CLASSES: frozenset[int] = C_MINOR_PITCH_CLASSES
    NOTE_NAMES: tuple[str, ...] = NOTE_NAMES
    COLOR_PITCH_MAP: dict[str, tuple[int, ...]] = COLOR_PITCH_MAP
    COLOR_CHANNEL_PROGRAM: dict[str, tuple[int, int]] = COLOR_CHANNEL_PROGRAM
    VERIDIS_QUO_PROGRESSION = VERIDIS_QUO_PROGRESSION
    HARMONIC_PROGRESSION = HARMONIC_PROGRESSION

    PAD_CHANNEL: int = PAD_CHANNEL
    BASS_CHANNEL: int = BASS_CHANNEL
    ARP_CHANNEL: int = ARP_CHANNEL
    DRUM_CHANNEL: int = DRUM_CHANNEL
    WARM_PAD_CHANNEL: int = WARM_PAD_CHANNEL
    PLUCK_CHANNEL: int = PLUCK_CHANNEL
    CHURCH_ORGAN_CHANNEL: int = CHURCH_ORGAN_CHANNEL

    DRUM_KICK: int = DRUM_KICK
    DRUM_RIM: int = DRUM_RIM
    DRUM_CLAP: int = DRUM_CLAP
    DRUM_HAT_CLOSED: int = DRUM_HAT_CLOSED
    DRUM_LOW_TOM: int = DRUM_LOW_TOM
    DRUM_HAT_OPEN: int = DRUM_HAT_OPEN
    DRUM_TAMBOURINE: int = DRUM_TAMBOURINE
    DRUM_CONGA_HI: int = DRUM_CONGA_HI
    DRUM_SHAKER: int = DRUM_SHAKER

    snap_to_c_minor = staticmethod(snap_to_c_minor)
    midi_to_note_name = staticmethod(midi_to_note_name)
    midi_to_freq = staticmethod(midi_to_freq)
    select_rhythm = staticmethod(select_rhythm)
    should_trigger_data_step = staticmethod(should_trigger_data_step)
    intensity_to_velocity = staticmethod(intensity_to_velocity)
    intensity_to_volume = staticmethod(intensity_to_volume)
    plan_musical_step = staticmethod(plan_musical_step)
    plan_ambient_drums = staticmethod(plan_ambient_drums)

    def __init__(
        self,
        backend: Optional[AudioBackend] = None,
        enable_audio: bool = True,
        ambient_drums: bool = True,
    ) -> None:
        self.backend = backend or create_best_audio_backend(enable_audio=enable_audio)
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self._ambient_drums_enabled = ambient_drums
        self._latest_analysis = AnalysisResult(
            dominant_color="Blue",
            intensity=0.0,
            density=0.0,
            shape="uniform",
            shape_score=0.0,
            spottiness=0.0,
            swirliness=0.0,
            sensor_size=40,
            sensor_bbox=(0, 0, 0, 0),
            cursor_xy=(0, 0),
            touching=False,
        )
        self._active_timed_notes: list[tuple[int, int, int]] = []  # (ch, note, off_step)
        self._active_pad_notes: list[tuple[int, int]] = []
        self._active_bass_notes: list[tuple[int, int]] = []
        self._active_drum_notes: list[int] = []

        self._pending_events: list[tuple[int, NoteEvent]] = []  # delayed earcon notes
        self._mix_state: dict = {"touching": False}
        self._beat_info = (0, 0)  # (bar_0..7, beat_0..3)
        self._current_telemetry = "75 BPM | Standby"
        self._melodic_channels: tuple[int, ...] = ()

    @property
    def ambient_drums_enabled(self) -> bool:
        with self._lock:
            return self._ambient_drums_enabled

    def toggle_drums(self) -> bool:
        """Toggle the Channel 9 ambient drum & percussion groove."""
        with self._lock:
            self._ambient_drums_enabled = not self._ambient_drums_enabled
            return self._ambient_drums_enabled

    def current_beat_info(self) -> tuple[int, int]:
        """Return ``(bar_index 0..7, beat_in_bar 0..3)`` for live UI beat indicator."""
        with self._lock:
            return self._beat_info

    def start(self) -> None:
        """Initialize the audio backend, configure spatial CCs, and start clock thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()

        channel_programs: dict[int, int] = {
            self.PAD_CHANNEL: 16,           # Drawbar Organ
            self.BASS_CHANNEL: 38,          # Synth Bass 1
            self.ARP_CHANNEL: 80,           # Lead 1 (Square)
            self.WARM_PAD_CHANNEL: 89,      # Warm Pad
            self.PLUCK_CHANNEL: 12,         # Marimba
            self.CHURCH_ORGAN_CHANNEL: 19,  # Church Organ
        }
        for channel, program in self.COLOR_CHANNEL_PROGRAM.values():
            channel_programs[channel] = program

        self._melodic_channels = tuple(
            sorted(ch for ch in channel_programs if ch != self.DRUM_CHANNEL)
        )
        self.backend.start(channel_programs)

        # Initial Veridis Quo warm reverb & chorus spatial setup
        for ch in set(channel_programs) | {self.DRUM_CHANNEL}:
            if ch in (self.PAD_CHANNEL, self.WARM_PAD_CHANNEL):
                reverb = 82
            elif ch == self.DRUM_CHANNEL:
                reverb = 42
            else:
                reverb = 68
            self.backend.control_change(ch, 91, reverb)
            if ch != self.DRUM_CHANNEL:
                self.backend.control_change(ch, 93, 35)
            # Mix balance + stereo placement so the color voice sits on top of the bed
            self.backend.control_change(ch, 7, CHANNEL_VOLUME.get(ch, 100))
            self.backend.control_change(ch, 10, int(round(64 + CHANNEL_PAN.get(ch, 0.0) * 50)))

        self._thread = threading.Thread(
            target=self._audio_loop,
            name="CMinorAudioEngine",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop the sequencer thread, silence all voices, and close the backend."""
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._silence_all_active()
        self.backend.close()

    def update_analysis(self, result: AnalysisResult) -> None:
        """Update the target CV metrics consumed on the 75 BPM beat grid."""
        with self._lock:
            self._latest_analysis = result

    def telemetry_text(self) -> str:
        """Return a thread-safe snapshot of the current musical status."""
        with self._lock:
            return self._current_telemetry

    def _silence_timed_notes(self) -> None:
        for ch, note, _off_step in self._active_timed_notes:
            self.backend.note_off(ch, note)
        self._active_timed_notes.clear()

    def _silence_pad_notes(self) -> None:
        for ch, note in self._active_pad_notes:
            self.backend.note_off(ch, note)
        self._active_pad_notes.clear()

    def _silence_bass_notes(self) -> None:
        for ch, note in self._active_bass_notes:
            self.backend.note_off(ch, note)
        self._active_bass_notes.clear()

    def _silence_drum_notes(self) -> None:
        for drum_note in self._active_drum_notes:
            self.backend.note_off(self.DRUM_CHANNEL, drum_note)
        self._active_drum_notes.clear()

    def _silence_all_active(self) -> None:
        self._silence_timed_notes()
        self._silence_pad_notes()
        self._silence_bass_notes()
        self._silence_drum_notes()

    def _release_expired_notes(self, global_step: int) -> None:
        still_active: list[tuple[int, int, int]] = []
        for ch, note, off_step in self._active_timed_notes:
            if global_step >= off_step:
                self.backend.note_off(ch, note)
            else:
                still_active.append((ch, note, off_step))
        self._active_timed_notes = still_active

    def _apply_expressive_ccs(
        self,
        analysis: AnalysisResult,
        step_16: int,
        last_cc_state: Optional[tuple[int, int, int, int]],
    ) -> tuple[int, int, int, int]:
        ccs = compute_expression_ccs(analysis, step_16)
        state = (ccs["reverb"], ccs["chorus"], ccs["modulation"], ccs["brightness"])

        if state != last_cc_state:
            for ch in self._melodic_channels:
                self.backend.control_change(ch, 91, ccs["reverb"])
                self.backend.control_change(ch, 93, ccs["chorus"])
                self.backend.control_change(ch, 1, ccs["modulation"])
                self.backend.control_change(ch, 74, ccs["brightness"])
                self.backend.control_change(ch, 11, ccs["expression"])

        swirl = float(getattr(analysis, "swirliness", 0.0))
        if swirl >= 0.15:
            self.backend.pitch_bend(self.ARP_CHANNEL, ccs["pitch_bend"])
        elif last_cc_state is not None and last_cc_state[2] > 0:
            self.backend.pitch_bend(self.ARP_CHANNEL, 8192)

        return state

    @staticmethod
    def _sensor_tier(sensor_size: int) -> str:
        if sensor_size < 30:
            return "tight"
        if sensor_size > 60:
            return "wide"
        return "mid"

    def mix_state(self) -> dict:
        """Thread-safe snapshot of what is playing right now (for the UI / status API)."""
        with self._lock:
            return dict(self._mix_state)

    def _release_channel_timed(self, channel: int, note: Optional[int] = None) -> None:
        kept: list[tuple[int, int, int]] = []
        for ch, n, off in self._active_timed_notes:
            if ch == channel and (note is None or n == note):
                self.backend.note_off(ch, n)
            else:
                kept.append((ch, n, off))
        self._active_timed_notes = kept

    def _emit(self, ev: NoteEvent, global_step: int) -> None:
        if ev.role == "drum":
            self.backend.note_on(self.DRUM_CHANNEL, ev.note, ev.velocity)
            self._active_drum_notes.append(ev.note)
            return
        if ev.role in PAD_ROLES:
            self.backend.note_on(ev.channel, ev.note, ev.velocity)
            self._active_pad_notes.append((ev.channel, ev.note))
            return
        self._release_channel_timed(ev.channel, ev.note)   # never stack the same pitch
        self.backend.note_on(ev.channel, ev.note, ev.velocity)
        off_step = global_step + max(1, int(round(ev.steps)))
        self._active_timed_notes.append((ev.channel, ev.note, off_step))

    def _dispatch(self, plan: StepPlan, global_step: int) -> None:
        # delayed notes (color-change earcon second note) that are now due
        due = [p for p in self._pending_events if p[0] <= global_step]
        self._pending_events = [p for p in self._pending_events if p[0] > global_step]
        for _start, ev in due:
            self._emit(ev, global_step)

        if plan.retrigger_pad:
            self._silence_pad_notes()
        if any(ev.role == "bass" for ev in plan.events):
            self._release_channel_timed(self.BASS_CHANNEL)      # mono bass
        for ev in plan.events:
            if ev.delay_steps > 0:
                self._pending_events.append((global_step + ev.delay_steps, ev))
            else:
                self._emit(ev, global_step)

    def _audio_loop(self) -> None:
        clock_origin = time.monotonic()
        global_step = 0
        last_cc_state: Optional[tuple[int, int, int, int]] = None
        sequencer = StepSequencer()
        smoother = AnalysisSmoother()

        while not self._stop_event.is_set():
            target_time = clock_origin + global_step * self.SIXTEENTH_DURATION_SEC
            now = time.monotonic()
            wait_sec = target_time - now
            if wait_sec > 0.002:
                self._stop_event.wait(timeout=wait_sec)
                if self._stop_event.is_set():
                    break
            elif wait_sec < -self.BEAT_DURATION_SEC:
                clock_origin = now - global_step * self.SIXTEENTH_DURATION_SEC

            step_16 = global_step % 16
            bar_index = (global_step // 16) % len(self.VERIDIS_QUO_PROGRESSION)
            beat_in_bar = step_16 // 4

            with self._lock:
                raw_analysis = self._latest_analysis
                drums_on = self._ambient_drums_enabled
                self._beat_info = (bar_index, beat_in_bar)

            self._silence_drum_notes()
            self._release_expired_notes(global_step)

            # Smooth jittery CV features so the music changes on purpose, not by accident
            analysis = smoother.update(raw_analysis)
            plan = sequencer.plan(analysis, global_step, drums_on)

            if not plan.touching:
                if plan.released:
                    self._silence_all_active()
                    self._pending_events.clear()
                    last_cc_state = None
                with self._lock:
                    self._current_telemetry = f"{self.BPM:.0f} BPM | Standby (hover map or press P)"
                    self._mix_state = {"touching": False}
                global_step += 1
                continue

            last_cc_state = self._apply_expressive_ccs(analysis, step_16, last_cc_state)
            self._dispatch(plan, global_step)

            sensor_size = int(getattr(analysis, "sensor_size", 40))
            instrument = COLOR_INSTRUMENT_NAME.get(analysis.dominant_color, "pad")
            drum_tag = "Drums ON" if drums_on else "Drums OFF"
            with self._lock:
                self._current_telemetry = (
                    f"{self.BPM:.0f} BPM | {plan.chord_name} ({bar_index + 1}/8) | "
                    f"{plan.tier_name}: {plan.rhythm_name} | "
                    f"{analysis.dominant_color}->{instrument} {plan.note_label} "
                    f"({getattr(analysis, 'shape', 'uniform')}) | Box {sensor_size}px | {drum_tag}"
                )
                self._mix_state = {
                    "touching": True,
                    "chord": plan.chord_name,
                    "bar": bar_index,
                    "beat": beat_in_bar,
                    "tier": plan.tier,
                    "tier_name": plan.tier_name,
                    "rhythm": plan.rhythm_name,
                    "color": analysis.dominant_color,
                    "instrument": instrument,
                    "note": plan.note_label,
                    "layers": active_layers(analysis, drums_on),
                }

            global_step += 1