{
  "state": {
    "plan": {
      "goal": "The amber beacon is lit.",
      "preconditions": [
        "The socket is connected to a working power source."
      ]
    },
    "observation": {
      "summary": "The socket is unplugged. The beacon is dark."
    }
  },
  "questions": {
    "condition": {
      "type": "choice",
      "instructions": "Using only the evidence in state.observation, determine whether state.plan.preconditions are supported. Select supported if every precondition is established, contradicted if any precondition is contradicted, or unknown if neither conclusion is established.",
      "criteria": {
        "supported": "Every required precondition is supported by the observation.",
        "contradicted": "The observation contradicts at least one required precondition.",
        "unknown": "No precondition is explicitly contradicted, but the observation does not establish every required precondition."
      }
    },
    "progress": {
      "type": "score",
      "instructions": "Using only state.observation, assess visible completion of state.plan.goal on the ordered levels below. Assess the goal itself independently of whether its preconditions hold. Do not infer unobserved progress.",
      "criteria": [
        "No visible progress toward the goal.",
        "Visible partial progress toward the goal, but completion is not established.",
        "The observation visibly establishes complete achievement of the goal."
      ]
    },
    "finished": {
      "type": "noul",
      "instructions": "Does state.observation explicitly establish that state.plan.goal is complete? Judge only the observed goal status; do not infer completion from preconditions or other question answers.",
      "criteria": {
        "true": "The observation explicitly establishes completion of the goal.",
        "false": "The observation does not explicitly establish completion of the goal."
      }
    }
  },
  "read_fields": {
    "condition": ["value", "probabilities", "confidence"],
    "progress": ["value"],
    "finished": ["value"]
  }
}