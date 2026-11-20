#!/usr/bin/env bash
# Make the Prefect stack start on boot, keep restarting, and start any
# unit that is not running. Units that are already running are not restarted.
# Run with sudo: sudo ops/systemd/install.sh
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
UNITS=/etc/systemd/system

# Served flows to keep up. Keep in sync with the watchdog config's systemd_units.
FLOW_UNITS=(
  prefect_onnx_inference
  prefect_slim_features
  prefect_slim_feature_validation
  prefect_yolo_inference
  prefect_yolo_training
  prefect_stingray_frame_timestamps
  prefect_stingray_image_abundance
  prefect_ifcb_flow_metric_evaluation
  prefect_ifcb_flow_metric_hyperparameter_search
  prefect_ifcb_flow_metric_inference
  prefect_ifcb_flow_metric_training
  prefect_ifcb_zipping_service
  prefect_image_classifier_dojo
)

if [[ $EUID -ne 0 ]]; then
  echo "Run with sudo." >&2
  exit 1
fi

install_dropin() {
  local unit=$1 src=$2
  install -d -m 0755 "$UNITS/$unit.service.d"
  install -m 0644 "$src" "$UNITS/$unit.service.d/10-amplify-robust.conf"
  echo "installed drop-in for $unit"
}

install_dropin prefect_server "$HERE/dropins/prefect-server.conf"
for unit in "${FLOW_UNITS[@]}"; do
  install_dropin "$unit" "$HERE/dropins/prefect-flow.conf"
done

# Old link to a unit that no longer exists.
link="$UNITS/multi-user.target.wants/prefect-db.service"
if [[ -L $link && ! -e $link ]]; then
  rm "$link"
  echo "removed dangling $link"
fi

systemctl daemon-reload
systemctl enable --now docker.service prefect_postgres prefect_server "${FLOW_UNITS[@]}"

echo
systemctl show prefect_postgres prefect_server "${FLOW_UNITS[@]}" \
  -p Id -p UnitFileState -p ActiveState -p Restart -p NRestarts | paste - - - - - - 
