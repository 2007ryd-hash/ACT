"""Export only reviewed summaries and selected numeric columns; never opens devices."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean

JOINTS = ('shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    out = Path(__file__).resolve().parents[1]
    for name in ('data', 'evidence', 'configs', 'assets'):
        (out / name).mkdir(exist_ok=True)
    sources = {}

    def source(relative):
        path = workspace / relative
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
        sources[relative] = {'sha256': digest.hexdigest(), 'bytes': path.stat().st_size}
        return path

    def read(relative):
        return json.loads(source(relative).read_text())

    def write(relative, value):
        (out / relative).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')

    def csvwrite(relative, fields, rows):
        with (out / relative).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    old = 'action复现外部视角，标记版/outputs/act-baseline-v1-100k/'
    rows = list(csv.DictReader(source(old + 'validation-comparison/extended-160k/checkpoint-comparison.csv').open()))
    hold = read('act-reproduction/evaluation/day5-20260923/hold-baseline-and-data-audit.json')
    csvwrite('data/historical-validation.csv', ['step', 'episode_macro_normalized_l1', 'hold_reference'], [
        dict(step=row['step'], episode_macro_normalized_l1=row['episode_macro_normalized_l1'],
             hold_reference=hold['hold_baseline']['episode_macro_normalized_l1']) for row in rows])
    historical_root = 'act-reproduction/evaluation/block-bag-v2-deployment-preflight-20261010/'
    comparison = read(historical_root + 'two-trial-comparison/comparison.json')
    historical = []
    for number, run in enumerate(comparison['runs'], 1):
        p = source(historical_root + 'te-trial/' + run['run'] + '/steps.jsonl')
        assert sources[str(p.relative_to(workspace))]['sha256'] == run['source_SHA256']['steps.jsonl']
        for line in p.open():
            row = json.loads(line)
            if row['sent']:
                historical.append(dict(trial=number, time_seconds=row['time_seconds'],
                                       te_target=row['TE_action'][5], measured_state=row['state'][5]))
    assert len(historical) == 3574
    csvwrite('data/historical-gripper.csv', ['trial', 'time_seconds', 'te_target', 'measured_state'], historical)
    write('evidence/historical-summary.json', {
        'frozen_validation': {'episodes': 10, 'frames': 17433, 'metric': 'episode-macro normalized L1',
                              'hold_reference': hold['hold_baseline']['episode_macro_normalized_l1']},
        'single_camera_trials': [dict(trial_id=r['run'], sent_actions=r['sent'],
                                     duration_s=r['duration_s'], gripper=r['gripper'],
                                     outcome=r['human_outcome']) for r in comparison['runs']],
        'not_a_pooled_success_rate': True,
    })
    offline = read('act-reproduction/evaluation/dual-validation5-finetune30k-20261026/comparison.json')
    write('evidence/offline-comparison.json', {
        'date': '2026-10-26', 'episodes': offline['episodes'], 'frames': offline['frames'],
        'old_model': 'dual-camera-50-demo-100k', 'new_model': 'dual-camera-70-demo-weighted-finetune-30k',
        'metrics': offline['overall'], 'per_episode_first_action': offline['per_episode_first_action'],
        'units': 'SO101 calibrated joint units, not degrees',
        'inputs': 'expert-trajectory front/wrist images and measured state; no expert action input',
        'TE_evaluated': False, 'closed_loop_success_measured': False,
        'identical_evaluator_and_validation_inputs': offline['identical_evaluator_and_validation_inputs'],
        'equivalent_processors': offline['equivalent_processors'],
    })
    session_path = 'act-reproduction/evaluation/dual-live-five-20261026/session-20261026-140625/session.json'
    session = read(session_path)
    checkpoint = Path(session['model']).relative_to(workspace)
    config = read(str(checkpoint / 'config.json'))
    write('configs/act-100k-model.json', config)
    model_sha = {Path(p).name: sha for p, sha in session['frozen_sha256'].items()
                 if Path(p).parent == Path(session['model'])}
    for path, expected in session['frozen_sha256'].items():
        rel = str(Path(path).relative_to(workspace))
        source(rel)
        assert sources[rel]['sha256'] == expected, rel
    success = []
    visual = {
        1: {'release_interval_s': [8.991, 9.252], 'release_frame_indices': [242, 249],
            'grasp_interval_s': [4, 5.5], 'lift_transport_interval_s': [5.5, 8]},
        5: {'release_interval_s': [8.748, 9.002], 'release_frame_indices': [237, 244],
            'grasp_interval_s': [4, 5.3], 'lift_transport_interval_s': [5.3, 8]},
    }
    for trial in session['trials']:
        if trial.get('result') != 'success':
            continue
        assert trial['intervention'] == 'no'
        number = trial['number']
        assert number in visual
        root = Path(trial['trial_dir']).relative_to(workspace)
        p = source(str(root / 'policy/steps.jsonl'))
        video = source(str(root / 'policy/trial.avi'))
        assert sources[str(p.relative_to(workspace))]['sha256'] == trial['artifact_sha256']['policy/steps.jsonl']
        assert sources[str(video.relative_to(workspace))]['sha256'] == trial['artifact_sha256']['policy/trial.avi']
        selected = []
        all_rows = []
        for frame_index, line in enumerate(p.open()):
            row = json.loads(line)
            all_rows.append(row)
            if not (0 <= row['time_seconds'] <= 20 and row['sent']):
                continue
            item = dict(frame_index=frame_index, step=row['step'], time_seconds=row['time_seconds'],
                        sent=int(row['sent']), loop_ms=row['loop_ms'],
                        range_clip_magnitude=row['range_clip_magnitude'])
            for key in ('state', 'TE_action', 'sender_return_action'):
                item.update({f'{key}.{joint}': row[key][i] for i, joint in enumerate(JOINTS)})
            selected.append(item)
        assert len(selected) == {1: 539, 5: 545}[number]
        filename = f'data/success-{number:02d}-first20s.csv'
        csvwrite(filename, list(selected[0]), selected)
        times = [r['time_seconds'] for r in selected]
        phases = dict(visual[number], withdrawal_interval_s=[10.5, 12], result_observed_at_s=[15, 20],
                      method='manual visual review of sampled frames with denser release-window inspection; log clock',
                      precision='approximate intervals, not automatic event detection')
        success.append({
            'case_id': f'success-{number:02d}', 'trial_id': root.name,
            'date': '2026-10-26', 'timezone': 'Asia/Shanghai', 'position_observer': trial['position'],
            'task': 'pick up the blue block, place it inside the black bag, release and withdraw',
            'task_outcome_observer': 'success', 'intervention_observer': 'no',
            'completion_time_s_observer_estimate': trial['completion_time_s_observer'],
            'video_review': phases, 'csv': filename,
            'export_window_s': [0, 20], 'exported_rows': len(selected),
            'first_last_log_time_s': [times[0], times[-1]],
            'mean_record_interval_ms': mean(b-a for a, b in zip(times, times[1:])) * 1000,
            'max_loop_ms_in_export': max(r['loop_ms'] for r in selected),
            'extra_range_clip_count_in_export': sum(r['range_clip_magnitude'] > 0 for r in selected),
            'initial_measured_state': all_rows[0]['state'],
            'joint_order': JOINTS,
            'source_steps': str(p.relative_to(workspace)),
            'source_video': str(video.relative_to(workspace)),
            'source_steps_sha256': trial['artifact_sha256']['policy/steps.jsonl'],
            'source_video_sha256': trial['artifact_sha256']['policy/trial.avi'],
            'video_frame_count': len(all_rows), 'video_header_fps': 30,
            'frame_mapping': 'video frame i corresponds to source JSONL row i; use time_seconds for wall-clock intervals',
            'scope': 'selected successful task segment; not the full-run lifecycle or a success-rate denominator',
        })
    assert len(success) == 2
    write('evidence/success-cases.json', {
        'selection': 'two explicitly labelled successful cases, not all attempts',
        'model': 'dual-camera-50-demo-100k', 'checkpoint_relative_to_workspace': str(checkpoint),
        'checkpoint_file_sha256': model_sha, 'deployment_settings': session['settings'],
        'joint_value_units': 'calibrated RANGE_M100_100 body / RANGE_0_100 gripper',
        'state_semantics': 'measured before this action; not the achieved state after the current command',
        'sender_return_action_semantics': 'LeRobot sender return, not exact encoded target or actual achieved position',
        'cases': success,
    })
    write('configs/experiment.json', {
        'environment': {'python': '3.11.13', 'pytorch': '2.7.0+cu128', 'lerobot': '0.4.3',
                        'lerobot_revision': 'e670ac5daf9b76cf5c18f632b89a8ff85c727ca6',
                        'image_id': 'sha256:8d6b3d2b76e9a61c37008f2ea0323e7551cc4375d227a6157b4cd1a3df7a097c',
                        'GPU': 'NVIDIA GeForce RTX 5060 Laptop, about 8 GiB'},
        'camera_mapping': {'front': 'SHUNCCM external camera', 'wrist': 'icSpring wrist camera'},
        'camera_config': {'width': 640, 'height': 480, 'fps_nominal': 30, 'input_mode': 'MJPG', 'model_color': 'RGB'},
        'training': {'episodes': 50, 'frames': 24118, 'steps': 100000, 'batch_size': 4,
                     'precision': 'FP32', 'seed': 1000, 'optimizer': 'AdamW', 'lr': 1e-5,
                     'chunk_size': 100, 'n_action_steps': 100, 'TE': None,
                     'initialization': 'new ACT parameters, ImageNet ResNet18 backbone',
                     'validation': '5 external episodes / 2337 frames, excluded from training and statistics'},
        'successful_deployment': session['settings'],
        'deployment_settings_are_historical_record_not_a_safe_motion_command': True,
        'later_finetune': {'episodes': 70, 'frames': 39468, 'additional_steps': 30000,
                           'old_frame_weight': 1, 'new_frame_weight': 2,
                           'normalization': 'baseline processors retained',
                           'success_cases_belong_to_this_model': False},
    })
    for relative in [
        'act-reproduction/notes/2026-09-30-experiment-failure-audit.md',
        'act-reproduction/notes/2026-09-23-last25-act-retrain.md',
        'act-reproduction/notes/2026-09-26-raw57-live-inference-contrast.md',
        'action复现外部视角，标记版/outputs/act-current-scene-three-model-evaluation/model-comparison.md',
        'DATA EXPERIMENT/three-target-57/model-version-map-20260927.md',
        'action-reproduction-wrist/records/wrist-act-te-trial-2026-09-30.md',
        'action-reproduction-wrist/records/wrist-act-stall-10-limit-2026-09-30.md',
        'act-reproduction/notes/2026-10-10-act-block-bag-v2-all50-100k-completion.md',
        'act-reproduction/training/dual-block-bag-all50-20261025/formal100k-20261025-230639-87674/completion-report.md',
    ]:
        source(relative)
    write('evidence/source-index.json', {'scope': 'Paths relative to the private local learning workspace; source files are not included.',
                                       'sources': sources})
    print(f'Exported {len(success)} successful cases, 1084 selected rows, 3574 historical rows; {len(sources)} source identities.')


if __name__ == '__main__':
    main()
