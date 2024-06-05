
#/bin/bash
set -eo pipefail

run_t3000_falcon7b_tests() {
  # Record the start time
  start_time=$(date +%s)

  echo "LOG_METAL: Running run_t3000_falcon7b_tests"

  env WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest models/demos/falcon7b/tests -m "model_perf_t3000"

  # Record the end time
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "LOG_METAL: run_t3000_falcon7b_tests $duration seconds to complete"
}

run_t3000_mixtral_tests() {
  # Record the start time
  start_time=$(date +%s)

  echo "LOG_METAL: Running run_t3000_mixtral_tests"

  env pytest models/demos/t3000/mixtral8x7b/tests/test_mixtral_perf.py::test_mixtral_model_perf[wormhole_b0-True-2048-150-0.025] -m "model_perf_t3000"

  # Record the end time
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "LOG_METAL: run_t3000_mixtral_tests $duration seconds to complete"
}

run_t3000_llama2_70b_tests() {
  # Record the start time
  start_time=$(date +%s)

  echo "LOG_METAL: Running run_t3000_llama2_70b_tests"

  env WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest models/experimental/llama2_70b/tests/test_llama_perf_decode.py -m "model_perf_t3000"

  # Record the end time
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "LOG_METAL: run_t3000_llama2_70b_tests $duration seconds to complete"
}

run_t3000_llm_tests() {
  # Run falcon7b tests
  run_t3000_falcon7b_tests

  # Run mixtral tests
  run_t3000_mixtral_tests

  # Run llama2-70b tests
  run_t3000_llama2_70b_tests

  # Merge all the generated reports
  env python models/perf/merge_perf_results.py
}

run_t3000_cnn_tests() {
  # Merge all the generated reports
  env python models/perf/merge_perf_results.py
}

main() {
  # Parse the arguments
  while [[ $# -gt 0 ]]; do
    case $1 in
      --pipeline-type)
        pipeline_type=$2
        shift
        ;;
      *)
        echo "Unknown option: $1"
        exit 1
        ;;
    esac
    shift
  done

  if [[ -z "$TT_METAL_HOME" ]]; then
    echo "Must provide TT_METAL_HOME in environment" 1>&2
    exit 1
  fi

  if [[ -z "$ARCH_NAME" ]]; then
    echo "Must provide ARCH_NAME in environment" 1>&2
    exit 1
  fi

  if [[ -z "$pipeline_type" ]]; then
    echo "--pipeline-type cannot be empty" 1>&2
    exit 1
  fi

  # Run all tests
  cd $TT_METAL_HOME
  export PYTHONPATH=$TT_METAL_HOME

  if [[ "$pipeline_type" == "llm_model_perf_t3000_device" ]]; then
    run_t3000_llm_tests
  elif [[ "$pipeline_type" == "cnn_model_perf_t3000_device" ]]; then
    run_t3000_cnn_tests
  else
    echo "$pipeline_type is invalid (supported: [cnn_model_perf_t3000_device, cnn_model_perf_t3000_device])" 2>&1
    exit 1
  fi
}

main "$@"
