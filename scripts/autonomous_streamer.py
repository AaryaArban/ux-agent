import os
import time
import json
import random
import yaml

# 🔧 Load Config to find the monitor directory
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

MONITOR_DIR = config['autonomy']['monitor_dir']
os.makedirs(MONITOR_DIR, exist_ok=True)

def generate_random_session():
    """Simulates different types of users with authentic, descriptive reviews."""
    rand = random.random()
    
    if rand > 0.7:  # 🟢 30% Happy Users
        happy_reviews = [
            "This app has completely changed how I manage my daily tasks; the UI is incredibly intuitive and snappy.",
            "I'm really impressed with the latest update, the navigation feels much smoother than the previous version.",
            "Absolutely love the dark mode implementation! It's clear the developers put a lot of thought into the user experience.",
            "Five stars! It's rare to find an app that is both feature-rich and this easy to navigate without a tutorial.",
            "The performance is rock solid. I haven't experienced a single lag spike since I started using it last week.",
            "Great job on the new checkout flow, it's much faster now and doesn't require unnecessary clicks.",
            "This is hands down the best app in its category. The transitions are fluid and the design is very modern.",
            "I love how the app anticipates what I need next; the predictive features are actually helpful for once.",
            "Everything works exactly as expected. It's refreshing to use a piece of software that just works perfectly.",
            "The haptic feedback and loading animations make the whole experience feel premium and high-end.",
            "Finally an app that doesn't drain my battery while providing such a smooth and responsive interface!",
            "Super satisfied with the speed of this tool. It saves me at least twenty minutes of work every single day.",
            "The integration with my other services is seamless. I didn't hit a single roadblock during the setup phase.",
            "I've recommended this to all my colleagues. The reliability and clean aesthetic are simply unmatched.",
            "Brilliant execution on the user interface; it's very accessible and everything is exactly where it should be."
        ]
        data = {
            "logs": [random.uniform(0.1, 1.0) for _ in range(10)],
            "behavior": [0, 0, 0, random.randint(0, 1), 0],
            "review_text": random.choice(happy_reviews)
        }
        
    elif rand > 0.3:  # 🟡 40% Neutral Users
        neutral_reviews = [
            "The app is decent for basic use, but I find the menu layout a bit confusing at times.",
            "It gets the job done, though I noticed some minor stuttering when scrolling through long lists.",
            "An average experience overall. It works well enough, but there isn't anything particularly special about it.",
            "I like the features provided, but the loading times could definitely be improved in the next update.",
            "The app is okay, but I've seen better designs elsewhere. It's functional but lacks that 'wow' factor.",
            "It's a solid app, but it crashed once when I was trying to upload a large file earlier today.",
            "I'm satisfied for now, but I hope they fix the occasional freezing that happens on the home screen.",
            "The interface is fine, but some of the buttons are a bit too small for easy tapping on my device.",
            "It's useful for what I need, but the navigation feels a little dated compared to other modern apps.",
            "A decent effort, but I feel like some of the advanced features are hidden too deep in the settings.",
            "The app performs okay most of the time, though it occasionally feels a bit sluggish under heavy use.",
            "I have mixed feelings; the core functionality is great but the secondary features feel half-baked.",
            "It's fine for occasional use, but I wouldn't rely on it for anything mission-critical just yet.",
            "The recent update fixed some bugs, but it also introduced a bit of lag that wasn't there before.",
            "Good app overall, but the notification system is a bit overwhelming and hard to customize."
        ]
        data = {
            "logs": [random.uniform(1.5, 3.5) for _ in range(10)],
            "behavior": [random.randint(0, 2) for _ in range(5)],
            "review_text": random.choice(neutral_reviews)
        }
        
    else:  # 🔴 30% Frustrated Users
        frustrated_reviews = [
            "I am extremely disappointed. The app keeps crashing every time I try to open the main dashboard.",
            "This is unusable! The lag is so bad that it takes five seconds just to register a single button press.",
            "I've lost my data twice now because the app closed unexpectedly. I would not recommend this to anyone.",
            "The UI is a disaster. I can't find anything and the constant loading spinners are driving me crazy.",
            "Worst update ever. Everything was working fine until yesterday, now it's just a buggy mess.",
            "I'm tired of the 'Something went wrong' errors. Why can't this app just maintain a stable connection?",
            "The navigation is a labyrinth. I spent ten minutes just trying to find the settings page. Terrible UX.",
            "Avoid this app at all costs. It's slow, bloated, and crashes at least three times every hour.",
            "I keep clicking buttons and nothing happens. It's like the app isn't even registering my inputs correctly.",
            "The performance is abysmal. It feels like I'm running this on a ten-year-old phone even though I have a flagship.",
            "Total waste of time. The app froze during a critical task and I had to restart my phone to get it working again.",
            "I hate how complicated they made the login process. It's like they want to make it hard for users to get in.",
            "The constant stuttering and visual glitches make the app feel very cheap and poorly developed.",
            "Nothing works! I've tried reinstalling three times but the crashing issue still persists on the home screen.",
            "I'm uninstalling this immediately. The level of frustration this app causes is simply not worth the features."
        ]
        data = {
            "logs": [random.uniform(4.0, 7.0) for _ in range(10)],
            "behavior": [random.randint(3, 8) for _ in range(5)],
            "review_text": random.choice(frustrated_reviews)
        }
    return data

if __name__ == "__main__":
    print(f"🚀 Starting Autonomous User Streamer with Authentic Reviews...")
    print(f"📡 Sending data to: {MONITOR_DIR}")
    
    session_count = 1
    try:
        while True:
            session_data = generate_random_session()
            file_name = f"session_{int(time.time())}_{session_count}.json"
            file_path = os.path.join(MONITOR_DIR, file_name)
            
            with open(file_path, "w") as f:
                json.dump(session_data, f)
            
            print(f"📤 [Session {session_count}] Generated & Streamed.")
            session_count += 1
            
            # Wait 5-15 seconds between 'users' to simulate real traffic
            time.sleep(random.randint(5, 15))
            
    except KeyboardInterrupt:
        print("\n🛑 Streamer stopped.")