# 2026-06-05T03:27:02.749411200
import vitis

client = vitis.create_client()
client.set_workspace(path="FPGA")

comp = client.create_hls_component(name = "mitchel_sub",cfg_file = ["hls_config.cfg"],template = "empty_hls_component")

comp = client.get_component(name="mitchel_sub")
comp.run(operation="C_SIMULATION")

vitis.dispose()

