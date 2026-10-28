"""Flower engine: the same client/server logic (fedcore) inside Flower's Simulation Engine (Ray).

Uses Flower's legacy Strategy path because SecAggPlusWorkflow requires a LegacyContext. With secure aggregation
the strategy only receives the unmasked aggregate, exactly what fedcore.ServerLogic expects for one unit.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from burstguard.engines.common import JsonlWriter, auto_workers, slim_eval, summarize

_WORKER = None


def _worker(cfg):
    global _WORKER
    if _WORKER is None:
        from burstguard.fedcore import SiteWorker
        from burstguard.logutil import quiet_logging
        quiet_logging()  # runs inside the Ray actor process
        _WORKER = SiteWorker(cfg)
    return _WORKER


def make_client_app(cfg: dict[str, Any]):
    from flwr.client import ClientApp, NumPyClient
    from flwr.client.mod import secaggplus_mod

    from burstguard import model as M
    from burstguard.attacks.client import attack_active, select_malicious

    malicious = set(select_malicious(cfg))

    class SiteClient(NumPyClient):
        def __init__(self, pid: int):
            self.pid = pid

        def fit(self, parameters, config):
            import traceback
            try:
                w = _worker(cfg)
                rnd = int(config.get("server_round", 0))
                role = "malicious" if (self.pid in malicious and attack_active(cfg, rnd)) else "benign"
                out = w.run({"site": self.pid, "round": rnd, "theta": M.join_arrays(parameters), "role": role})
                info = {k: v for k, v in out["info"].items() if isinstance(v, (int, float, str, bool))}
                info["pid"] = self.pid
                return M.split_vec(out["theta"], w.layout), 1, info
            except Exception:
                print(f"[client {self.pid}] fit failed:\n{traceback.format_exc()}", flush=True)
                raise

    def client_fn(context):
        return SiteClient(int(context.node_config["partition-id"])).to_client()

    mods = [secaggplus_mod] if cfg["secagg"]["enabled"] else None
    return ClientApp(client_fn=client_fn, mods=mods)


def make_server_app(cfg: dict[str, Any], run_dir: Path, log=print):
    from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
    from flwr.server import LegacyContext, ServerApp, ServerConfig
    from flwr.server.strategy import FedAvg
    from flwr.server.workflow import DefaultWorkflow, SecAggPlusWorkflow

    from burstguard import model as M
    from burstguard.fedcore import ServerLogic

    write = JsonlWriter(run_dir / "metrics.jsonl")
    server = ServerLogic(cfg, log=log)
    R, n = int(cfg["fl"]["rounds"]), int(cfg["data"]["num_sites"])
    every = max(1, int(cfg["fl"]["eval_every"]))
    state = {"theta": server.theta0.copy(), "t": time.time()}
    nbytes = int(server.theta0.nbytes)
    maxabs = float(np.abs(server.theta0).max())
    write("setup", model=server.model_info.__dict__, n_trainable=int(server.theta0.size), update_bytes=nbytes,
          malicious=server.malicious, benign=server.benign, engine="flower", theta_max_abs=maxabs)
    if cfg["secagg"]["enabled"] and maxabs > float(cfg["secagg"]["clipping_range"]):
        log(f"[server] WARNING: max |theta| = {maxabs:.2f} exceeds secagg.clipping_range; values will be clipped")

    class Strategy(FedAvg):
        def configure_fit(self, server_round, parameters, client_manager):
            state["t"] = time.time()
            return super().configure_fit(server_round, parameters, client_manager)

        def aggregate_fit(self, server_round, results, failures):
            if failures:
                msgs = [repr(f)[:2000] for f in failures]
                write("client_failures", round=server_round, n=len(failures), details=msgs)
                raise RuntimeError(f"{len(failures)} client(s) failed in round {server_round}: {msgs[0]}")
            infos = [dict(r.metrics) for _, r in results]
            res = [{"site": int(i["pid"]), "theta": None, "info": i} for i in infos]
            secure = None
            if cfg["secagg"]["enabled"]:
                secure = M.join_arrays(parameters_to_ndarrays(results[0][1].parameters))
            else:
                for (_, r), item in zip(results, res):
                    item["theta"] = M.join_arrays(parameters_to_ndarrays(r.parameters))
            res.sort(key=lambda x: x["site"])
            theta_new, rec = server.aggregate(server_round, state["theta"], res, secure_mean_theta=secure)
            state["theta"] = theta_new
            rec.update(round_wall_s=round(time.time() - state["t"], 3), update_bytes=nbytes, clients=infos)
            write("round", **rec)
            log(f"[server] round {server_round}/{R}: accepted={rec['accepted']} "
                f"loss={np.mean([i['train_loss'] for i in infos]):.4f} {rec['round_wall_s']:.1f}s")
            return ndarrays_to_parameters(M.split_vec(theta_new, server.layout)), {}

    def evaluate_fn(server_round, parameters, config):
        if not (server_round == 0 or server_round % every == 0 or server_round == R):
            return None
        theta = M.join_arrays(parameters)
        splits = [cfg["fl"]["eval_split"]] + (list(cfg["fl"]["final_splits"]) if server_round == R else [])
        msgs = []
        for sp in splits:
            if sp not in server.evaluator.data:
                continue
            ev = server.evaluate(theta, sp)
            write("eval", round=server_round, split=sp, **slim_eval(ev), per_site=ev["sites"])
            msgs.append(f"{sp}: MASE={ev.get('mase_daily_mean', float('nan')):.3f} ASR={ev.get('asr_mean', float('nan')):.3f}")
        log(f"[server] eval round {server_round}: " + "  ".join(msgs))
        return 0.0, {}

    app = ServerApp()

    @app.main()
    def main(grid, context):
        strategy = Strategy(fraction_fit=1.0, fraction_evaluate=0.0, min_fit_clients=n, min_available_clients=n,
                            initial_parameters=ndarrays_to_parameters(M.split_vec(server.theta0, server.layout)),
                            evaluate_fn=evaluate_fn, on_fit_config_fn=lambda r: {"server_round": r},
                            accept_failures=True)
        ctx = LegacyContext(context=context, config=ServerConfig(num_rounds=R), strategy=strategy)
        sa = cfg["secagg"]
        fit_wf = SecAggPlusWorkflow(num_shares=int(sa["num_shares"]),
                                    reconstruction_threshold=int(sa["reconstruction_threshold"]),
                                    max_weight=1.0, clipping_range=float(sa["clipping_range"]),
                                    quantization_range=int(sa["quantization_range"])) if sa["enabled"] else None
        DefaultWorkflow(fit_workflow=fit_wf)(grid, ctx)
        write("finished", rounds=R)
        (run_dir / "server_finished").write_text("ok\n")

    return app, server


def run_flower(cfg: dict[str, Any], run_dir: Path, log=print) -> dict[str, Any]:
    from flwr.simulation import run_simulation
    n = int(cfg["data"]["num_sites"])
    ncpu = int(cfg["sim"]["num_cpus"]) or auto_workers(0, n)
    cpc = float(cfg["sim"]["cpus_per_client"])
    init_args: dict[str, Any] = {"num_cpus": ncpu, "include_dashboard": False, "log_to_driver": True,
                                 "address": "local"}
    if cfg["sim"].get("ray_temp_dir"):
        init_args["_temp_dir"] = str(cfg["sim"]["ray_temp_dir"])
    log(f"[flower] Ray local cluster: num_cpus={ncpu}, cpus_per_client={cpc}, {n} virtual SuperNodes")
    server_app, _ = make_server_app(cfg, run_dir, log)
    if cfg["model"]["source"] == "hf":
        os.environ["HF_HUB_OFFLINE"] = "1"   # workers read the cache populated by the server
    run_simulation(server_app=server_app, client_app=make_client_app(cfg), num_supernodes=n,
                   backend_config={"client_resources": {"num_cpus": cpc, "num_gpus": 0.0}, "init_args": init_args})
    if not (run_dir / "server_finished").exists():
        raise RuntimeError("ServerApp did not finish - see the traceback above in console.log")
    return summarize(run_dir)
