from locust import HttpUser, task, between, events
import random
import time

class RecruiterUser(HttpUser):
    # Simulate a user taking between 1 and 3 seconds between actions
    wait_time = between(1, 3)

    @task
    def rank_candidates(self):
        # Mock JD content
        jds = [
            {"jd_text": "Need a senior backend engineer with Python and Redis expertise.", "industry_type": "Technical"},
            {"jd_text": "Looking for an Engineering Manager to lead a team of 10.", "industry_type": "Leadership"},
            {"jd_text": "Need a freelance UI/UX designer for a 3-month project.", "industry_type": "Gig"}
        ]
        payload = random.choice(jds)

        start_time = time.time()
        
        # Post request to the rank endpoint
        with self.client.post("/rank", json=payload, catch_response=True) as response:
            total_time_ms = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                data = response.json()
                
                # We assume the API endpoint includes timing metadata in the response headers
                # or body for identifying bottlenecks (e.g., X-ANN-Time, X-Rerank-Time)
                # Here we mock parsing that out
                ann_time = response.headers.get("X-ANN-Time", "N/A")
                rerank_time = response.headers.get("X-Rerank-Time", "N/A")
                
                # Log the breakdown
                # In a real Locust scenario, you'd emit custom events or write to a logger
                # print(f"Latency Breakout | ANN: {ann_time}ms | Rerank: {rerank_time}ms | Total: {total_time_ms:.2f}ms")
                
                # Custom assertion: Fail the request internally if latency > 500ms
                if total_time_ms > 500:
                    response.failure(f"Latency exceeded 500ms: {total_time_ms:.2f}ms")
                else:
                    response.success()
            else:
                response.failure(f"Failed with status {response.status_code}")

# Custom Event to check 95th percentile at the end of the run
@events.quitting.add_listener
def assert_95th_percentile(environment, **kw):
    if environment.stats.total.num_requests == 0:
        return
        
    p95_latency = environment.stats.total.get_response_time_percentile(0.95)
    
    print(f"\n--- Load Test Results ---")
    print(f"Total Requests: {environment.stats.total.num_requests}")
    print(f"95th Percentile Latency: {p95_latency}ms")
    
    if p95_latency > 500:
        print(f"[FAIL] 95th Percentile Latency ({p95_latency}ms) is strictly greater than 500ms limit.")
        environment.process_exit_code = 1
    else:
        print(f"[PASS] 95th Percentile Latency ({p95_latency}ms) is under the 500ms limit.")
