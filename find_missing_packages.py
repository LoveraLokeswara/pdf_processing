# Read the contents of all_packages.txt
with open('all_packages.txt', 'r') as f:
    all_packages = set(f.read().splitlines())

# Read the contents of requirements.txt
with open('requirements.txt', 'r') as f:
    requirements = set(line.split('==')[0].strip() for line in f if line.strip() and not line.startswith('#'))

# Find packages in all_packages.txt that are not in requirements.txt
missing_packages = all_packages - requirements

# Print the missing packages
for package in sorted(missing_packages):
    print(package)