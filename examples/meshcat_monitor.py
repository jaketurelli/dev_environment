
import pinocchio as pin
import time
import threading
import subprocess
import meshcat


class MonitorVisualizer():
    """
    Example usage:\n
        monitor = meshcat_monitor.MonitorVisualizer(meshcat_visualizer_object)\n

        # update simulation while the window is open\n
        while monitor.is_open():\n
            meshcat_visualizer_object.display( ... )\n
            time.sleep(0.01)\n
    """

    def __init__(self, viz_obj, block_until_first_open=True, is_open_poll_period=0.5, try_to_detect_tab_close=True):
        self._try_to_detect_tab_close = try_to_detect_tab_close
        self._is_open = False
        self._last_open = 0
        self._is_open_poll_period = is_open_poll_period
        self._monitor_thread = threading.Thread(target=self._monitor, daemon=True)
        self._monitor_thread.start()
        if not hasattr(viz_obj, 'window') and hasattr(viz_obj, 'viewer'):
            viz_obj = viz_obj.viewer
        if not hasattr(viz_obj, 'window'):
            raise TypeError(
                'MonitorVisualizer first argument must be of type "meshcat.Visualizer" or '
                f'"pinocchio.visualize.MeshcatVisualizer" not {type(viz_obj)}')
        self._port_str = ':' + viz_obj.window.web_url.rsplit(':', 1)[-1].split('/')[0] + ' (ESTABLISHED)'
        if block_until_first_open:
            while not self._is_open:
                time.sleep(self._is_open_poll_period)

    def is_open(self) -> bool:
        """returns true if the visualizer tab/window is still open"""
        return self._is_open

    def _monitor(self):
        while True:
            self._is_open = self._check_is_open()
            time.sleep(self._is_open_poll_period)

    def _check_is_open(self):
        lsof_out_str = subprocess.check_output(['lsof', '-i', '-P', '-n']).decode('utf-8')
        connections_established = lsof_out_str.count(self._port_str)
        if self._try_to_detect_tab_close:
            reduced_by = self._last_open - connections_established
            if reduced_by == 1:
                self._last_open = connections_established
                return False
        self._last_open = connections_established
        return connections_established > 0


class MeshcatVisualizer(pin.visualize.MeshcatVisualizer):
    """Same as Pinocchio.visualize.MeshcatVisualizer, but adds a is_open() method"""

    def __init__(
        self,
        model=pin.Model(),
        collision_model=None,
        visual_model=None,
        copy_models=False,
        data=None,
        collision_data=None,
        visual_data=None,
    ):
        super().__init__(model, collision_model, visual_model, copy_models, data, collision_data, visual_data)
        self._monitor = None

    def initViewer(self, viewer=None, open=False, loadModel=False, block_until_first_open=True, is_open_poll_period=0.5, try_to_detect_tab_close=True):
        """Start a new MeshCat server and client.
        Note: the server can also be started separately using the "meshcat-server" command in a terminal:
        this enables the server to remain active after the current script ends.
        """

        super().initViewer(viewer, open, loadModel)
        self._monitor = MonitorVisualizer(self, open and block_until_first_open, is_open_poll_period, try_to_detect_tab_close)

    def is_open(self) -> bool:
        """returns true if the visualizer tab/window is still open"""
        return self._monitor._is_open


if __name__ == "__main__":
    viz = MeshcatVisualizer()
    viz.initViewer(open=True)
    while viz.is_open():
        time.sleep(0.25)
