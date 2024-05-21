# Yolox_m ttnn implementation

## Current status

1. ttnn.conv in focus sub-module has PCC of 0.99. However, torch slice is used in this sub-module, because reference has this type of slice x[..., ::2, ::2] where ::2 indicates alternative elements will be chosen, and its throwing error in ::2.
2. Tested dark2 module getting pcc = 0.9852007334820249, however using one torch concat because ttnn.concat returning wrong shaped tensor.
3. Tested dark3 module getting pcc = 0.9804122271033656
4. Tested dark4 module getting pcc = 0.9824520887645896
5. Till dark4 integration getting pcc = 0.643640977409953
6. Currently working on dark5 testing. Also will work on integration pcc improvement
