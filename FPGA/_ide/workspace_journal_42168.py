# 2026-06-05T03:26:47.213186
import vitis

client = vitis.create_client()
client.set_workspace(path="FPGA")

vitis.dispose()

