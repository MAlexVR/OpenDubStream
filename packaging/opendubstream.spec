# Supply project_license only after the owner chooses the project-wide license.
%{!?project_license:%{error:Owner license decision required; define project_license only after approval}}
%global debug_package %{nil}
Name:           opendubstream
Version:        0.1.0~alpha.1
Release:        1%{?dist}
Summary:        Experimental local English-to-Spanish captions and dubbing
License:        %{project_license} AND CC0-1.0
URL:            https://github.com/MAlexVR/OpenDubStream
Source0:        opendubstream-%{version}.tar.gz
BuildArch:      noarch
Requires:       python3
Requires:       python3.12
Requires:       bash
Requires:       pipewire-utils
Requires:       pulseaudio-utils
Requires:       pipewire-pulseaudio
Requires:       espeak-ng
Requires:       libxkbcommon-x11
Requires:       xcb-util-cursor
Requires:       mesa-libGL
Requires:       fontconfig

%description
Experimental desktop captions and dubbing for a selected Chrome audio stream.
Includes the application and an explicit per-user setup launcher. Model weights
and Python inference dependencies are NOT bundled. Run opendubstream setup in
a desktop terminal and review its download forecast before opting in.
Fedora 44 x86_64 with Python 3.12 and configured NVIDIA CUDA is the alpha target.
Continuous real-time operation is not guaranteed.

%prep
%setup -q

%build
# Pure Python/source payload. No downloads or environment bootstrap at build time.

%install
mkdir -p %{buildroot}%{_datadir}/opendubstream
cp -a src assets scripts requirements LICENSE LICENSES THIRD_PARTY_NOTICES.md %{buildroot}%{_datadir}/opendubstream/
install -Dm755 packaging/opendubstream.py %{buildroot}%{_bindir}/opendubstream
install -Dm644 assets/opendubstream.desktop %{buildroot}%{_datadir}/applications/opendubstream.desktop
install -Dm644 assets/opendubstream.svg %{buildroot}%{_datadir}/icons/hicolor/scalable/apps/opendubstream.svg

%files
%doc README.md docs/installation.md docs/troubleshooting.md
%license LICENSE THIRD_PARTY_NOTICES.md LICENSES/BluCast-MIT.txt assets/LICENSES/CC0-1.0.txt
%{_bindir}/opendubstream
%{_datadir}/opendubstream
%{_datadir}/applications/opendubstream.desktop
%{_datadir}/icons/hicolor/scalable/apps/opendubstream.svg

%changelog
* Tue Sep 08 2026 Mauricio Vargas - 0.1.0~alpha.1-1
- Package the experimental desktop application with explicit per-user setup.
