"""Run the real integration suite with Linux kernel socket creation denied."""
import ctypes as C
import os
from pathlib import Path
import sys

# x86_64 Linux seccomp BPF: socket(2) -> EACCES; everything else allowed.
# NO_NEW_PRIVS permits this without root/CAP_SYS_ADMIN. Filters survive exec/fork,
# so both this process and all native workers cannot open any network socket.
class Filter(C.Structure):
    _fields_=[('code',C.c_ushort),('jt',C.c_ubyte),('jf',C.c_ubyte),('k',C.c_uint)]
class Program(C.Structure):
    _fields_=[('len',C.c_ushort),('filter',C.POINTER(Filter))]
filters=(Filter*4)(Filter(0x20,0,0,0),Filter(0x15,0,1,41),Filter(0x06,0,0,0x00050000|13),Filter(0x06,0,0,0x7fff0000))
libc=C.CDLL(None,use_errno=True)
if libc.prctl(38,1,0,0,0)!=0 or libc.prctl(22,2,C.byref(Program(4,filters)),0,0)!=0:
    raise OSError(C.get_errno(),'Could not apply offline verification filter')
import socket
try:socket.socket()
except PermissionError:pass
else:raise AssertionError('Network isolation failed')
os.environ.pop('DISPLAY',None)
os.environ['YGO_TEST_SOCKET_DENIED']='1'
root=Path(__file__).resolve().parent
os.execv(sys.executable,[sys.executable,str(root/'verify_suite.py')])
