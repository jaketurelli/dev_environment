
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

    def __init__(self, viz_obj: meshcat.Visualizer, wait_for_first_open=True, poll_period=0.25):
        self._is_open = False
        self._poll_period = poll_period
        self._monitor_thread = threading.Thread(target=self._monitor)
        self._monitor_thread.setDaemon(True)
        self._monitor_thread.start()
        if not hasattr(viz_obj, 'window') and hasattr(viz_obj, 'viewer'):
            viz_obj = viz_obj.viewer
        if not hasattr(viz_obj, 'window'):
            raise TypeError(
                f'MonitorVisualizer first argument must be of type "meshcat.Visualizer" or "pinocchio.visualize.MeshcatVisualizer" not {type(viz_obj)}')
        self._port_str = ':' + viz_obj.window.web_url.rsplit(':', 1)[-1].split('/')[0] + ' (ESTABLISHED)'
        if wait_for_first_open:
            while not self._is_open:
                time.sleep(self._poll_period)

    def is_open(self) -> bool:
        return self._is_open

    def _monitor(self):
        while True:
            self._is_open = self._check_is_open()
            time.sleep(self._poll_period)

    def _check_is_open(self):
        lsof_out_str = subprocess.check_output(['lsof', '-i', '-P', '-n']).decode('utf-8')
        connections_established = lsof_out_str.count(self._port_str)
        return connections_established > 0
