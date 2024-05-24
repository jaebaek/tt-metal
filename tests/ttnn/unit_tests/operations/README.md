This commit contains conv unit test for segformer model.

1. `test_segformer_conv_gs_groups` method in `tests/ttnn/unit_tests/operations/test_conv2d.py` file will contain unit test for conv which has group>1. To run the test use the following command : `pytest tests/ttnn/unit_tests/operations/test_conv2d.py::test_segformer_conv_gs_groups`

2. `test_conv_segformer` method in `tests/ttnn/unit_tests/operations/test_conv2d.py` file will contain unit test for conv with groups=1, the one which is enabled is the failing test case. To run the test use the following command : `pytest tests/ttnn/unit_tests/operations/test_conv2d.py::test_conv_segformer`

Note: The unit-test are only for image resolution 512x512
